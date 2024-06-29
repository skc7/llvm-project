//===-- AMDGPUAsanInstrumentLDS.cpp
//-----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass instruments the LDS accesses lowered by amdgpu-sw-lower-lds pass to
// detect the addressing errors. It is implemented in two phases.
// Pass first updates the metadata global initializer of LDS to update the
// redzone sizes. It then instruments all the addrspace(3) accesses in the IR to
// add error detection logic to report addressing errors.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUTargetMachine.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/ReplaceConstant.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/OptimizedStructLayout.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Instrumentation/AddressSanitizer.h"
#include "llvm/Transforms/Instrumentation/AddressSanitizerCommon.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

#define DEBUG_TYPE "amdgpu-asan-instrument-lds"

using namespace llvm;
using namespace AMDGPU;

namespace {

const char kAMDGPUBallotName[] = "llvm.amdgcn.ballot.i64";
const char kAMDGPUUnreachableName[] = "llvm.amdgcn.unreachable";
const char kAMDGPULDSKernelId[] = "llvm.amdgcn.lds.kernel.id";

static const uint64_t kSmallX86_64ShadowOffsetBase = 0x7FFFFFFF;
static const uint64_t kSmallX86_64ShadowOffsetAlignMask = ~0xFFFULL;

static uint64_t getRedzoneSizeForScale(int AsanScale) {
  // Redzone used for stack and globals is at least 32 bytes.
  // For scales 6 and 7, the redzone has to be 64 and 128 bytes respectively.
  return std::max(32U, 1U << AsanScale);
}

static uint64_t getMinRedzoneSizeForGlobal(int AsanScale) {
  return getRedzoneSizeForScale(AsanScale);
}

static uint64_t getRedzoneSizeForGlobal(int AsanScale, uint64_t SizeInBytes) {
  constexpr uint64_t kMaxRZ = 1 << 18;
  const uint64_t MinRZ = getMinRedzoneSizeForGlobal(AsanScale);

  uint64_t RZ = 0;
  if (SizeInBytes <= MinRZ / 2) {
    // Reduce redzone size for small size objects, e.g. int, char[1]. MinRZ is
    // at least 32 bytes, optimize when SizeInBytes is less than or equal to
    // half of MinRZ.
    RZ = MinRZ - SizeInBytes;
  } else {
    // Calculate RZ, where MinRZ <= RZ <= MaxRZ, and RZ ~ 1/4 * SizeInBytes.
    RZ = std::clamp((SizeInBytes / MinRZ / 4) * MinRZ, MinRZ, kMaxRZ);

    // Round up to multiple of MinRZ.
    if (SizeInBytes % MinRZ)
      RZ += MinRZ - (SizeInBytes % MinRZ);
  }

  assert((RZ + SizeInBytes) % MinRZ == 0);

  return RZ;
}

static size_t TypeStoreSizeToSizeIndex(uint32_t TypeSize) {
  size_t Res = llvm::countr_zero(TypeSize / 8);
  return Res;
}

class AMDGPUAsanInstrumentLDS {
public:
  AMDGPUAsanInstrumentLDS(Module &Mod, const AMDGPUTargetMachine &TM)
      : M(Mod), AMDGPUTM(TM), IRB(Mod.getContext()) {}
  bool run();

private:
  bool preProcessAMDGPULDSAccesses(int AsanScale);
  bool poisonRedzonesForSwLDS(Function &F,
                              SmallVector<std::pair<uint32_t, uint32_t>, 64>
                                  &RedzoneOffsetAndSizeVector);

  bool instrumentLDSAccesses(int AsanScale, int AsanOffset);
  SmallVector<std::pair<uint32_t, uint32_t>, 64>
  updateSwLDSMetadataWithRedzoneInfo(Function &F, int Scale);
  void recordLDSAbsoluteAddress(GlobalVariable *GV, uint32_t Address);
  void updateLDSSizeFnAttr(Function *Func, uint32_t Offset, bool UsesDynLDS);
  GlobalVariable *getKernelSwLDSMetadataGlobal(Function &F);
  GlobalVariable *getKernelSwDynLDSGlobal(Function &F);
  GlobalVariable *getKernelSwLDSGlobal(Function &F);
  GlobalVariable *getKernelSwLDSBaseGlobal();

  void instrumentAddress(Instruction *OrigIns, Instruction *InsertBefore,
                         Value *Addr, MaybeAlign Alignment,
                         uint32_t TypeStoreSize, bool IsWrite,
                         Value *SizeArgument, bool UseCalls, bool Recover,
                         int AsanScale, int AsanOffset);

  Value *memToShadow(Value *Shadow, int AsanScale, uint32_t AsanOffset);

  Instruction *generateCrashCode(Instruction *InsertBefore, Value *Addr,
                                 bool IsWrite, size_t AccessSizeIndex,
                                 Value *SizeArgument, bool Recover);

  Value *createSlowPathCmp(Value *AddrLong, Value *ShadowValue,
                           uint32_t TypeStoreSize, int AsanScale);

  Instruction *genAMDGPUReportBlock(Value *Cond, bool Recover);

  void getInterestingMemoryOperands(
      Instruction *I, SmallVectorImpl<InterestingMemoryOperand> &Interesting);

  Module &M;
  const AMDGPUTargetMachine &AMDGPUTM;
  IRBuilder<> IRB;
};

Instruction *AMDGPUAsanInstrumentLDS::genAMDGPUReportBlock(Value *Cond,
                                                           bool Recover) {
  Value *ReportCond = Cond;
  if (!Recover) {
    auto Ballot = M.getOrInsertFunction(kAMDGPUBallotName, IRB.getInt64Ty(),
                                        IRB.getInt1Ty());
    ReportCond = IRB.CreateIsNotNull(IRB.CreateCall(Ballot, {Cond}));
  }

  auto *Trm = SplitBlockAndInsertIfThen(
      ReportCond, &*IRB.GetInsertPoint(), false,
      MDBuilder(M.getContext()).createBranchWeights(1, 100000));
  Trm->getParent()->setName("asan.report");

  if (Recover)
    return Trm;

  Trm = SplitBlockAndInsertIfThen(Cond, Trm, false);
  IRB.SetInsertPoint(Trm);
  return IRB.CreateCall(
      M.getOrInsertFunction(kAMDGPUUnreachableName, IRB.getVoidTy()), {});
}

Value *AMDGPUAsanInstrumentLDS::createSlowPathCmp(Value *AddrLong,
                                                  Value *ShadowValue,
                                                  uint32_t TypeStoreSize,
                                                  int AsanScale) {

  unsigned int LongSize = M.getDataLayout().getPointerSizeInBits();
  IntegerType *IntptrTy = Type::getIntNTy(M.getContext(), LongSize);
  size_t Granularity = static_cast<size_t>(1) << AsanScale;
  // Addr & (Granularity - 1)
  Value *LastAccessedByte =
      IRB.CreateAnd(AddrLong, ConstantInt::get(IntptrTy, Granularity - 1));
  // (Addr & (Granularity - 1)) + size - 1
  if (TypeStoreSize / 8 > 1)
    LastAccessedByte = IRB.CreateAdd(
        LastAccessedByte, ConstantInt::get(IntptrTy, TypeStoreSize / 8 - 1));
  // (uint8_t) ((Addr & (Granularity-1)) + size - 1)
  LastAccessedByte =
      IRB.CreateIntCast(LastAccessedByte, ShadowValue->getType(), false);
  // ((uint8_t) ((Addr & (Granularity-1)) + size - 1)) >= ShadowValue
  return IRB.CreateICmpSGE(LastAccessedByte, ShadowValue);
}

Instruction *AMDGPUAsanInstrumentLDS::generateCrashCode(
    Instruction *InsertBefore, Value *Addr, bool IsWrite,
    size_t AccessSizeIndex, Value *SizeArgument, bool Recover) {
  IRB.SetInsertPoint(InsertBefore);
  CallInst *Call = nullptr;
  int LongSize = M.getDataLayout().getPointerSizeInBits();
  Type *IntptrTy = Type::getIntNTy(M.getContext(), LongSize);
  const char kAsanReportErrorTemplate[] = "__asan_report_";
  const std::string TypeStr = IsWrite ? "store" : "load";
  const std::string EndingStr = Recover ? "_noabort" : "";
  SmallVector<Type *, 3> Args2 = {IntptrTy, IntptrTy};
  AttributeList AL2;
  FunctionCallee AsanErrorCallbackSized = M.getOrInsertFunction(
      kAsanReportErrorTemplate + TypeStr + "_n" + EndingStr,
      FunctionType::get(IRB.getVoidTy(), Args2, false), AL2);
  const std::string Suffix = TypeStr + llvm::itostr(1ULL << AccessSizeIndex);
  SmallVector<Type *, 2> Args1{1, IntptrTy};
  AttributeList AL1;
  FunctionCallee AsanErrorCallback = M.getOrInsertFunction(
      kAsanReportErrorTemplate + Suffix + EndingStr,
      FunctionType::get(IRB.getVoidTy(), Args1, false), AL1);
  if (SizeArgument) {
    Call = IRB.CreateCall(AsanErrorCallbackSized, {Addr, SizeArgument});
  } else {
    Call = IRB.CreateCall(AsanErrorCallback, Addr);
  }

  Call->setCannotMerge();
  return Call;
}

Value *AMDGPUAsanInstrumentLDS::memToShadow(Value *Shadow, int AsanScale,
                                            uint32_t AsanOffset) {
  int LongSize = M.getDataLayout().getPointerSizeInBits();
  Type *IntptrTy = Type::getIntNTy(M.getContext(), LongSize);
  // Shadow >> scale
  Shadow = IRB.CreateLShr(Shadow, AsanScale);
  if (AsanOffset == 0)
    return Shadow;
  // (Shadow >> scale) | offset
  Value *ShadowBase = ConstantInt::get(IntptrTy, AsanOffset);
  return IRB.CreateAdd(Shadow, ShadowBase);
}

void AMDGPUAsanInstrumentLDS::instrumentAddress(
    Instruction *OrigIns, Instruction *InsertBefore, Value *Addr,
    MaybeAlign Alignment, uint32_t TypeStoreSize, bool IsWrite,
    Value *SizeArgument, bool UseCalls, bool Recover, int AsanScale,
    int AsanOffset) {
  int LongSize = M.getDataLayout().getPointerSizeInBits();
  Type *IntptrTy = Type::getIntNTy(M.getContext(), LongSize);
  IRB.SetInsertPoint(InsertBefore);
  size_t AccessSizeIndex = TypeStoreSizeToSizeIndex(TypeStoreSize);
  Type *ShadowTy = IntegerType::get(M.getContext(),
                                    std::max(8U, TypeStoreSize >> AsanScale));
  Type *ShadowPtrTy = PointerType::get(ShadowTy, 0);
  Value *AddrLong;
  Type *PtrTy = cast<PointerType>(Addr->getType()->getScalarType());
  if (PtrTy->getPointerAddressSpace() == 3) {
    Function *Func = IRB.GetInsertBlock()->getParent();
    Value *SwLDS;
    if (Func->getCallingConv() == CallingConv::AMDGPU_KERNEL) {
      SwLDS = getKernelSwLDSGlobal(*Func);
    } else {
      GlobalVariable *LDSBaseTable = getKernelSwLDSBaseGlobal();
      if (LDSBaseTable) {
        auto *KernelId = IRB.CreateCall(
            M.getOrInsertFunction(kAMDGPULDSKernelId, IRB.getInt32Ty()), {});
        Value *BaseGEP =
            IRB.CreateInBoundsGEP(LDSBaseTable->getValueType(), LDSBaseTable,
                                  {IRB.getInt32(0), KernelId});
        SwLDS = IRB.CreateLoad(IRB.getPtrTy(3), BaseGEP);
      } else {
        SwLDS = IRB.CreateIntToPtr(IRB.getInt32(0), IRB.getPtrTy(3));
      }
    }
    assert(SwLDS && "Invalid AMDGPU Sw LDS base ptr");
    Value *PtrToInt = IRB.CreatePtrToInt(Addr, IRB.getInt32Ty());
    Value *LoadMallocPtr = IRB.CreateLoad(IRB.getPtrTy(1), SwLDS);
    Value *GEP =
        IRB.CreateInBoundsGEP(IRB.getInt8Ty(), LoadMallocPtr, {PtrToInt});
    AddrLong = IRB.CreatePointerCast(GEP, IntptrTy);
  } else
    AddrLong = IRB.CreatePointerCast(Addr, IntptrTy);
  Value *ShadowPtr = memToShadow(AddrLong, AsanScale, AsanOffset);
  const uint64_t ShadowAlign =
      std::max<uint64_t>(Alignment.valueOrOne().value() >> AsanScale, 1);
  Value *ShadowValue = IRB.CreateAlignedLoad(
      ShadowTy, IRB.CreateIntToPtr(ShadowPtr, ShadowPtrTy), Align(ShadowAlign));
  Value *Cmp = IRB.CreateIsNotNull(ShadowValue);
  auto *Cmp2 =
      createSlowPathCmp(AddrLong, ShadowValue, TypeStoreSize, AsanScale);
  Cmp = IRB.CreateAnd(Cmp, Cmp2);
  Instruction *CrashTerm = genAMDGPUReportBlock(Cmp, Recover);
  Instruction *Crash = generateCrashCode(
      CrashTerm, AddrLong, IsWrite, AccessSizeIndex, SizeArgument, Recover);
  if (OrigIns->getDebugLoc())
    Crash->setDebugLoc(OrigIns->getDebugLoc());
  return;
}

GlobalVariable *
AMDGPUAsanInstrumentLDS::getKernelSwLDSMetadataGlobal(Function &F) {
  SmallString<64> KernelLDSName("llvm.amdgcn.sw.lds.");
  KernelLDSName += F.getName();
  KernelLDSName += ".md";
  return M.getNamedGlobal(KernelLDSName);
}

GlobalVariable *AMDGPUAsanInstrumentLDS::getKernelSwDynLDSGlobal(Function &F) {
  SmallString<64> KernelLDSName("llvm.amdgcn.");
  KernelLDSName += F.getName();
  KernelLDSName += ".dynlds";
  return M.getNamedGlobal(KernelLDSName);
}

GlobalVariable *AMDGPUAsanInstrumentLDS::getKernelSwLDSGlobal(Function &F) {
  SmallString<64> KernelLDSName("llvm.amdgcn.sw.lds.");
  KernelLDSName += F.getName();
  return M.getNamedGlobal(KernelLDSName);
}

GlobalVariable *AMDGPUAsanInstrumentLDS::getKernelSwLDSBaseGlobal() {
  SmallString<64> KernelLDSName("llvm.amdgcn.sw.lds.base.table");
  return M.getNamedGlobal(KernelLDSName);
}

void AMDGPUAsanInstrumentLDS::updateLDSSizeFnAttr(Function *Func,
                                                  uint32_t Offset,
                                                  bool UsesDynLDS) {
  if (Offset != 0) {
    SmallString<256> Buffer;
    raw_svector_ostream SS(Buffer);
    SS << format("%u", Offset);
    if (UsesDynLDS)
      SS << format(",%u", Offset);
    Func->addFnAttr("amdgpu-lds-size", Buffer);
  }
}

void AMDGPUAsanInstrumentLDS::recordLDSAbsoluteAddress(GlobalVariable *GV,
                                                       uint32_t Address) {
  LLVMContext &Ctx = M.getContext();
  auto *IntTy = M.getDataLayout().getIntPtrType(Ctx, 3);
  auto *MinC = ConstantAsMetadata::get(ConstantInt::get(IntTy, Address));
  auto *MaxC = ConstantAsMetadata::get(ConstantInt::get(IntTy, Address + 1));
  GV->setMetadata(LLVMContext::MD_absolute_symbol,
                  MDNode::get(Ctx, {MinC, MaxC}));
}

/// Update SwLDS Metadata global initializer with redzone info.
/// Poison redzone regions using the redzone size and offset info.
bool AMDGPUAsanInstrumentLDS::preProcessAMDGPULDSAccesses(int AsanScale) {
  bool IsChanged = false;
  for (Function &F : M) {
    if (!F.hasFnAttribute(Attribute::SanitizeAddress))
      continue;
    auto RedzoneOffsetAndSizeVector =
        updateSwLDSMetadataWithRedzoneInfo(F, AsanScale);
    IsChanged |= poisonRedzonesForSwLDS(F, RedzoneOffsetAndSizeVector);
  }
  return IsChanged;
}

/// Update SwLDS Metadata global initializer with redzone info.
SmallVector<std::pair<uint32_t, uint32_t>, 64>
AMDGPUAsanInstrumentLDS::updateSwLDSMetadataWithRedzoneInfo(Function &F,
                                                            int Scale) {
  Module *M = F.getParent();
  GlobalVariable *SwLDSMetadataGlobal = getKernelSwLDSMetadataGlobal(F);
  GlobalVariable *SwLDSGlobal = getKernelSwLDSGlobal(F);
  if (!SwLDSMetadataGlobal || !SwLDSGlobal)
    return {};

  LLVMContext &Ctx = M->getContext();
  Type *Int32Ty = Type::getInt32Ty(Ctx);
  SmallVector<std::pair<uint32_t, uint32_t>, 64> RedzoneOffsetAndSizeVector;
  Constant *MdInit = SwLDSMetadataGlobal->getInitializer();
  Align LDSAlign = Align(SwLDSGlobal->getAlign().valueOrOne());

  StructType *MDStructType =
      cast<StructType>(SwLDSMetadataGlobal->getValueType());
  unsigned NumStructs = MDStructType->getNumElements();
  std::vector<Constant *> Initializers;
  uint32_t MallocSize = 0;
  StructType *LDSItemTy =
      cast<StructType>(MDStructType->getStructElementType(0));

  for (unsigned i = 0; i < NumStructs; i++) {
    ConstantStruct *member =
        dyn_cast<ConstantStruct>(MdInit->getAggregateElement(i));
    Constant *NewInitItem;
    if (member) {
      ConstantInt *GlobalSize =
          cast<ConstantInt>(member->getAggregateElement(1U));
      unsigned GlobalSizeValue = GlobalSize->getZExtValue();
      Constant *NewItemStartOffset = ConstantInt::get(Int32Ty, MallocSize);
      if (GlobalSizeValue) {
        Constant *NewItemGlobalSizeConst =
            ConstantInt::get(Int32Ty, GlobalSizeValue);
        const uint64_t RightRedzoneSize =
            getRedzoneSizeForGlobal(Scale, GlobalSizeValue);
        MallocSize += GlobalSizeValue;
        RedzoneOffsetAndSizeVector.emplace_back(MallocSize, RightRedzoneSize);
        MallocSize += RightRedzoneSize;
        unsigned NewItemAlignGlobalPlusRedzoneSize =
            alignTo(GlobalSizeValue + RightRedzoneSize, LDSAlign);
        Constant *NewItemAlignGlobalPlusRedzoneSizeConst =
            ConstantInt::get(Int32Ty, NewItemAlignGlobalPlusRedzoneSize);
        NewInitItem = ConstantStruct::get(
            LDSItemTy, {NewItemStartOffset, NewItemGlobalSizeConst,
                        NewItemAlignGlobalPlusRedzoneSizeConst});
        MallocSize = alignTo(MallocSize, LDSAlign);
      } else {
        Constant *CurrMallocSize = ConstantInt::get(Int32Ty, MallocSize);
        Constant *zero = ConstantInt::get(Int32Ty, 0);
        NewInitItem =
            ConstantStruct::get(LDSItemTy, {CurrMallocSize, zero, zero});
        RedzoneOffsetAndSizeVector.emplace_back(0, 0);
      }
    } else {
      Constant *CurrMallocSize = ConstantInt::get(Int32Ty, MallocSize);
      Constant *zero = ConstantInt::get(Int32Ty, 0);
      NewInitItem =
          ConstantStruct::get(LDSItemTy, {CurrMallocSize, zero, zero});
      RedzoneOffsetAndSizeVector.emplace_back(0, 0);
    }
    Initializers.push_back(NewInitItem);
  }
  GlobalVariable *SwDynLDS = getKernelSwDynLDSGlobal(F);
  bool usesDynLDS = SwDynLDS != nullptr;
  updateLDSSizeFnAttr(&F, MallocSize, usesDynLDS);
  if (usesDynLDS)
    recordLDSAbsoluteAddress(SwDynLDS, MallocSize);

  Constant *Data = ConstantStruct::get(MDStructType, Initializers);
  SwLDSMetadataGlobal->setInitializer(Data);
  return RedzoneOffsetAndSizeVector;
}

void AMDGPUAsanInstrumentLDS::getInterestingMemoryOperands(
    Instruction *I, SmallVectorImpl<InterestingMemoryOperand> &Interesting) {
  const DataLayout &DL = M.getDataLayout();
  unsigned int LongSize = M.getDataLayout().getPointerSizeInBits();
  Type *IntptrTy = Type::getIntNTy(M.getContext(), LongSize);
  if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
    Interesting.emplace_back(I, LI->getPointerOperandIndex(), false,
                             LI->getType(), LI->getAlign());
  } else if (StoreInst *SI = dyn_cast<StoreInst>(I)) {
    Interesting.emplace_back(I, SI->getPointerOperandIndex(), true,
                             SI->getValueOperand()->getType(), SI->getAlign());
  } else if (AtomicRMWInst *RMW = dyn_cast<AtomicRMWInst>(I)) {
    Interesting.emplace_back(I, RMW->getPointerOperandIndex(), true,
                             RMW->getValOperand()->getType(), std::nullopt);
  } else if (AtomicCmpXchgInst *XCHG = dyn_cast<AtomicCmpXchgInst>(I)) {
    Interesting.emplace_back(I, XCHG->getPointerOperandIndex(), true,
                             XCHG->getCompareOperand()->getType(),
                             std::nullopt);
  } else if (auto CI = dyn_cast<CallInst>(I)) {
    switch (CI->getIntrinsicID()) {
    case Intrinsic::masked_load:
    case Intrinsic::masked_store:
    case Intrinsic::masked_gather:
    case Intrinsic::masked_scatter: {
      bool IsWrite = CI->getType()->isVoidTy();
      // Masked store has an initial operand for the value.
      unsigned OpOffset = IsWrite ? 1 : 0;
      auto BasePtr = CI->getOperand(OpOffset);
      Type *Ty = IsWrite ? CI->getArgOperand(0)->getType() : CI->getType();
      MaybeAlign Alignment = Align(1);
      // Otherwise no alignment guarantees. We probably got Undef.
      if (auto *Op = dyn_cast<ConstantInt>(CI->getOperand(1 + OpOffset)))
        Alignment = Op->getMaybeAlignValue();
      Value *Mask = CI->getOperand(2 + OpOffset);
      Interesting.emplace_back(I, OpOffset, IsWrite, Ty, Alignment, Mask);
      break;
    }
    case Intrinsic::masked_expandload:
    case Intrinsic::masked_compressstore: {
      bool IsWrite = CI->getIntrinsicID() == Intrinsic::masked_compressstore;
      unsigned OpOffset = IsWrite ? 1 : 0;
      auto BasePtr = CI->getOperand(OpOffset);
      MaybeAlign Alignment = BasePtr->getPointerAlignment(DL);
      Type *Ty = IsWrite ? CI->getArgOperand(0)->getType() : CI->getType();
      IRBuilder<> IB(I);
      Value *Mask = CI->getOperand(1 + OpOffset);
      // Use the popcount of Mask as the effective vector length.
      Type *ExtTy = VectorType::get(IntptrTy, cast<VectorType>(Ty));
      Value *ExtMask = IB.CreateZExt(Mask, ExtTy);
      Value *EVL = IB.CreateAddReduce(ExtMask);
      Value *TrueMask = ConstantInt::get(Mask->getType(), 1);
      Interesting.emplace_back(I, OpOffset, IsWrite, Ty, Alignment, TrueMask,
                               EVL);
      break;
    }
    case Intrinsic::vp_load:
    case Intrinsic::vp_store:
    case Intrinsic::experimental_vp_strided_load:
    case Intrinsic::experimental_vp_strided_store: {
      auto *VPI = cast<VPIntrinsic>(CI);
      unsigned IID = CI->getIntrinsicID();
      bool IsWrite = CI->getType()->isVoidTy();
      unsigned PtrOpNo = *VPI->getMemoryPointerParamPos(IID);
      Type *Ty = IsWrite ? CI->getArgOperand(0)->getType() : CI->getType();
      MaybeAlign Alignment = VPI->getOperand(PtrOpNo)->getPointerAlignment(DL);
      Value *Stride = nullptr;
      if (IID == Intrinsic::experimental_vp_strided_store ||
          IID == Intrinsic::experimental_vp_strided_load) {
        Stride = VPI->getOperand(PtrOpNo + 1);
        // Use the pointer alignment as the element alignment if the stride is a
        // mutiple of the pointer alignment. Otherwise, the element alignment
        // should be Align(1).
        unsigned PointerAlign = Alignment.valueOrOne().value();
        if (!isa<ConstantInt>(Stride) ||
            cast<ConstantInt>(Stride)->getZExtValue() % PointerAlign != 0)
          Alignment = Align(1);
      }
      Interesting.emplace_back(I, PtrOpNo, IsWrite, Ty, Alignment,
                               VPI->getMaskParam(), VPI->getVectorLengthParam(),
                               Stride);
      break;
    }
    case Intrinsic::vp_gather:
    case Intrinsic::vp_scatter: {
      auto *VPI = cast<VPIntrinsic>(CI);
      unsigned IID = CI->getIntrinsicID();
      bool IsWrite = IID == Intrinsic::vp_scatter;
      unsigned PtrOpNo = *VPI->getMemoryPointerParamPos(IID);
      Type *Ty = IsWrite ? CI->getArgOperand(0)->getType() : CI->getType();
      MaybeAlign Alignment = VPI->getPointerAlignment();
      Interesting.emplace_back(I, PtrOpNo, IsWrite, Ty, Alignment,
                               VPI->getMaskParam(),
                               VPI->getVectorLengthParam());
      break;
    }
    default:
      for (unsigned ArgNo = 0; ArgNo < CI->arg_size(); ArgNo++) {
        if (!CI->isByValArgument(ArgNo))
          continue;
        Type *Ty = CI->getParamByValType(ArgNo);
        Interesting.emplace_back(I, ArgNo, false, Ty, Align(1));
      }
    }
  }
}

bool AMDGPUAsanInstrumentLDS::instrumentLDSAccesses(int AsanScale,
                                                    int AsanOffset) {
  bool IsChanged = false;
  SmallVector<InterestingMemoryOperand, 16> OperandsToInstrument;

  for (Function &F : M) {
    if (!F.hasFnAttribute(Attribute::SanitizeAddress))
      continue;
    for (auto &BB : F) {
      for (Instruction &Inst : BB) {
        SmallVector<InterestingMemoryOperand, 1> InterestingOperands;
        getInterestingMemoryOperands(&Inst, InterestingOperands);
        if (InterestingOperands.empty())
          continue;
        for (auto &Operand : InterestingOperands) {
          Value *Addr = Operand.getPtr();
          if (Addr->getType()->getPointerAddressSpace() !=
              AMDGPUAS::LOCAL_ADDRESS)
            continue;
          OperandsToInstrument.push_back(Operand);
        }
      }
    }
  }
  for (auto &Operand : OperandsToInstrument) {
    Value *Addr = Operand.getPtr();
    instrumentAddress(Operand.getInsn(), Operand.getInsn(), Addr,
                      Operand.Alignment, Operand.TypeStoreSize, Operand.IsWrite,
                      nullptr, false, false, AsanScale, AsanOffset);
    IsChanged = true;
  }
  return IsChanged;
}

/// Poison redzone regions using the redzone size and offset info.
bool AMDGPUAsanInstrumentLDS::poisonRedzonesForSwLDS(
    Function &F, SmallVector<std::pair<uint32_t, uint32_t>, 64>
                     &RedzoneOffsetAndSizeVector) {
  GlobalVariable *SwLDSGlobal = getKernelSwLDSGlobal(F);
  GlobalVariable *SwLDSMetadataGlobal = getKernelSwLDSMetadataGlobal(F);
  if (!SwLDSGlobal || !SwLDSMetadataGlobal)
    return false;

  LLVMContext &Ctx = M.getContext();
  Type *Int64Ty = Type::getInt64Ty(Ctx);
  Type *VoidTy = Type::getVoidTy(Ctx);
  FunctionCallee AsanPoisonRegion = M.getOrInsertFunction(
      StringRef("__asan_poison_region"),
      FunctionType::get(VoidTy, {Int64Ty, Int64Ty}, false));
  Constant *MdInit = SwLDSMetadataGlobal->getInitializer();

  for (User *U : SwLDSGlobal->users()) {
    StoreInst *SI = dyn_cast<StoreInst>(U);
    if (!SI)
      continue;

    Type *PtrTy =
        cast<PointerType>(SI->getValueOperand()->getType()->getScalarType());
    unsigned int AddrSpace = PtrTy->getPointerAddressSpace();
    if (AddrSpace != 1)
      report_fatal_error("AMDGPU illegal store to SW LDS");

    StructType *MDStructType =
        cast<StructType>(SwLDSMetadataGlobal->getValueType());
    unsigned NumStructs = MDStructType->getNumElements();
    Value *StoreMallocPointer = SI->getValueOperand();

    assert(RedzoneOffsetAndSizeVector.size() == NumStructs);
    for (unsigned i = 0; i < NumStructs; i++) {
      ConstantStruct *member =
          dyn_cast<ConstantStruct>(MdInit->getAggregateElement(i));
      if (!member)
        continue;

      ConstantInt *GlobalSize =
          cast<ConstantInt>(member->getAggregateElement(1U));
      unsigned GlobalSizeValue = GlobalSize->getZExtValue();

      if (!GlobalSizeValue)
        continue;
      IRB.SetInsertPoint(SI->getNextNode());
      auto &RedzonePair = RedzoneOffsetAndSizeVector[i];
      uint64_t RedzoneOffset = RedzonePair.first;
      uint64_t RedzoneSize = RedzonePair.second;

      Value *RedzoneAddrOffset = IRB.CreateInBoundsGEP(
          IRB.getInt8Ty(), StoreMallocPointer, {IRB.getInt64(RedzoneOffset)});
      Value *RedzoneAddress =
          IRB.CreatePtrToInt(RedzoneAddrOffset, IRB.getInt64Ty());
      IRB.CreateCall(AsanPoisonRegion,
                     {RedzoneAddress, IRB.getInt64(RedzoneSize)});
    }
  }
  return true;
}

bool AMDGPUAsanInstrumentLDS::run() {
  for (auto &GV : M.globals()) {
    if (GV.getType()->getPointerAddressSpace() != AMDGPUAS::LOCAL_ADDRESS)
      continue;
    assert(GV.isAbsoluteSymbolRef() && "Invalid LDS access to instrument");
  }
  bool IsChanged = false;
  uint64_t ShadowBase;
  int AsanScale;
  bool OrShadowOffset;
  unsigned LongSize = M.getDataLayout().getPointerSizeInBits();
  getAddressSanitizerParams(Triple(AMDGPUTM.getTargetTriple()), LongSize, false,
                            &ShadowBase, &AsanScale, &OrShadowOffset);
  uint32_t AsanOffset = (kSmallX86_64ShadowOffsetBase &
                         (kSmallX86_64ShadowOffsetAlignMask << AsanScale));

  IsChanged |= preProcessAMDGPULDSAccesses(AsanScale);
  IsChanged |= instrumentLDSAccesses(AsanScale, AsanOffset);

  return IsChanged;
}

class AMDGPUAsanInstrumentLDSLegacy : public ModulePass {
public:
  const AMDGPUTargetMachine *AMDGPUTM;
  static char ID;
  AMDGPUAsanInstrumentLDSLegacy(const AMDGPUTargetMachine *TM = nullptr)
      : ModulePass(ID), AMDGPUTM(TM) {
    initializeAMDGPUAsanInstrumentLDSLegacyPass(
        *PassRegistry::getPassRegistry());
  }
  bool runOnModule(Module &M) override;
};
} // namespace

char AMDGPUAsanInstrumentLDSLegacy::ID = 0;
char &llvm::AMDGPUAsanInstrumentLDSLegacyPassID =
    AMDGPUAsanInstrumentLDSLegacy::ID;

INITIALIZE_PASS_BEGIN(AMDGPUAsanInstrumentLDSLegacy,
                      "amdgpu-asan-instrument-lds",
                      "AMDGPU address sanitizer instrumentation of LDS", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(AMDGPUAsanInstrumentLDSLegacy, "amdgpu-asan-instrument-lds",
                    "AMDGPU address sanitizer instrumentation of LDS", false,
                    false)

bool AMDGPUAsanInstrumentLDSLegacy::runOnModule(Module &M) {
  if (!AMDGPUTM) {
    auto &TPC = getAnalysis<TargetPassConfig>();
    AMDGPUTM = &TPC.getTM<AMDGPUTargetMachine>();
  }
  AMDGPUAsanInstrumentLDS AsanInstrumentLDS(M, *AMDGPUTM);
  return AsanInstrumentLDS.run();
}

ModulePass *
llvm::createAMDGPUAsanInstrumentLDSLegacyPass(const AMDGPUTargetMachine *TM) {
  return new AMDGPUAsanInstrumentLDSLegacy(TM);
}

PreservedAnalyses AMDGPUAsanInstrumentLDSPass::run(Module &M,
                                                   ModuleAnalysisManager &AM) {
  AMDGPUAsanInstrumentLDS AsanInstrumentLDS(M, TM);
  bool IsChanged = AsanInstrumentLDS.run();
  if (!IsChanged)
    return PreservedAnalyses::all();
  return PreservedAnalyses::none();
}