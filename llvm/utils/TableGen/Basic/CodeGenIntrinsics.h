//===- CodeGenIntrinsics.h - Intrinsic Class Wrapper -----------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a wrapper class for the 'Intrinsic' TableGen class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_UTILS_TABLEGEN_BASIC_CODEGENINTRINSICS_H
#define LLVM_UTILS_TABLEGEN_BASIC_CODEGENINTRINSICS_H

#include "SDNodeProperties.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ModRef.h"
#include <string>
#include <tuple>
#include <vector>

namespace llvm {
class Record;
class RecordKeeper;

// Global information needed to build intrinsics.
struct CodeGenIntrinsicContext {
  explicit CodeGenIntrinsicContext(const RecordKeeper &RC);
  std::vector<const Record *> DefaultProperties;

  // Maximum number of values an intrinsic can return.
  unsigned MaxNumReturn;
};

struct CodeGenIntrinsic {
  const Record *TheDef; // The actual record defining this intrinsic.
  std::string Name;     // The name of the LLVM function "llvm.bswap.i32"
  StringRef EnumName;   // The name of the enum "bswap_i32"
  StringRef ClangBuiltinName; // Name of the corresponding GCC builtin, or "".
  StringRef MSBuiltinName;    // Name of the corresponding MS builtin, or "".
  StringRef TargetPrefix;     // Target prefix, e.g. "ppc" for t-s intrinsics.

  /// This structure holds the return values and parameter values of an
  /// intrinsic. If the number of return values is > 1, then the intrinsic
  /// implicitly returns a first-class aggregate. The numbering of the types
  /// starts at 0 with the first return value and continues from there through
  /// the parameter list. This is useful for "matching" types.
  struct IntrinsicSignature {
    /// The MVT::SimpleValueType for each return type. Note that this list is
    /// only populated when in the context of a target .td file. When building
    /// Intrinsics.td, this isn't available, because we don't know the target
    /// pointer size.
    std::vector<const Record *> RetTys;

    /// The MVT::SimpleValueType for each parameter type. Note that this list is
    /// only populated when in the context of a target .td file.  When building
    /// Intrinsics.td, this isn't available, because we don't know the target
    /// pointer size.
    std::vector<const Record *> ParamTys;
  };

  IntrinsicSignature IS;

  /// Memory effects of the intrinsic.
  MemoryEffects ME = MemoryEffects::unknown();

  /// SDPatternOperator Properties applied to the intrinsic.
  unsigned Properties = 0;

  /// This is set to true if the intrinsic is overloaded by its argument
  /// types.
  bool isOverloaded = false;

  /// True if the intrinsic is commutative.
  bool isCommutative = false;

  /// True if the intrinsic can throw.
  bool canThrow = false;

  /// True if the intrinsic is marked as noduplicate.
  bool isNoDuplicate = false;

  /// True if the intrinsic is marked as nomerge.
  bool isNoMerge = false;

  /// True if the intrinsic is no-return.
  bool isNoReturn = false;

  /// True if the intrinsic is no-callback.
  bool isNoCallback = false;

  /// True if the intrinsic is no-sync.
  bool isNoSync = false;

  /// True if the intrinsic is no-free.
  bool isNoFree = false;

  /// True if the intrinsic is will-return.
  bool isWillReturn = false;

  /// True if the intrinsic is cold.
  bool isCold = false;

  /// True if the intrinsic is marked as convergent.
  bool isConvergent = false;

  /// True if the intrinsic has side effects that aren't captured by any
  /// of the other flags.
  bool hasSideEffects = false;

  // True if the intrinsic is marked as speculatable.
  bool isSpeculatable = false;

  // True if the intrinsic is marked as strictfp.
  bool isStrictFP = false;

  enum ArgAttrKind {
    NoCapture,
    NoAlias,
    NoUndef,
    NonNull,
    Returned,
    ReadOnly,
    WriteOnly,
    ReadNone,
    ImmArg,
    Alignment,
    Dereferenceable,
    Range,
  };

  struct ArgAttribute {
    ArgAttrKind Kind;
    uint64_t Value;
    uint64_t Value2;

    ArgAttribute(ArgAttrKind K, uint64_t V, uint64_t V2)
        : Kind(K), Value(V), Value2(V2) {}

    bool operator<(const ArgAttribute &Other) const {
      return std::tie(Kind, Value, Value2) <
             std::tie(Other.Kind, Other.Value, Other.Value2);
    }
  };

  /// Vector of attributes for each argument.
  SmallVector<SmallVector<ArgAttribute, 0>> ArgumentAttributes;

  void addArgAttribute(unsigned Idx, ArgAttrKind AK, uint64_t V = 0,
                       uint64_t V2 = 0);

  bool hasProperty(enum SDNP Prop) const { return Properties & (1 << Prop); }

  /// Goes through all IntrProperties that have IsDefault value set and sets
  /// the property.
  void setDefaultProperties(ArrayRef<const Record *> DefaultProperties);

  /// Helper function to set property \p Name to true.
  void setProperty(const Record *R);

  /// Returns true if the parameter at \p ParamIdx is a pointer type. Returns
  /// false if the parameter is not a pointer, or \p ParamIdx is greater than
  /// the size of \p IS.ParamVTs.
  ///
  /// Note that this requires that \p IS.ParamVTs is available.
  bool isParamAPointer(unsigned ParamIdx) const;

  bool isParamImmArg(unsigned ParamIdx) const;

  CodeGenIntrinsic(const Record *R, const CodeGenIntrinsicContext &Ctx);
};

class CodeGenIntrinsicTable {
public:
  struct TargetSet {
    StringRef Name;
    size_t Offset;
    size_t Count;
  };

  explicit CodeGenIntrinsicTable(const RecordKeeper &RC);

  bool empty() const { return Intrinsics.empty(); }
  size_t size() const { return Intrinsics.size(); }
  auto begin() const { return Intrinsics.begin(); }
  auto end() const { return Intrinsics.end(); }
  const CodeGenIntrinsic &operator[](size_t Pos) const {
    return Intrinsics[Pos];
  }
  ArrayRef<CodeGenIntrinsic> operator[](const TargetSet &Set) const {
    return ArrayRef(&Intrinsics[Set.Offset], Set.Count);
  }
  ArrayRef<TargetSet> getTargets() const { return Targets; }

private:
  void CheckDuplicateIntrinsics() const;
  void CheckTargetIndependentIntrinsics() const;
  void CheckOverloadSuffixConflicts() const;

  std::vector<CodeGenIntrinsic> Intrinsics;
  std::vector<TargetSet> Targets;
};

// This class builds `CodeGenIntrinsic` on demand for a given Def.
class CodeGenIntrinsicMap {
  DenseMap<const Record *, std::unique_ptr<CodeGenIntrinsic>> Map;
  const CodeGenIntrinsicContext Ctx;

public:
  explicit CodeGenIntrinsicMap(const RecordKeeper &RC) : Ctx(RC) {}
  const CodeGenIntrinsic &operator[](const Record *Def);
};

} // namespace llvm

#endif // LLVM_UTILS_TABLEGEN_BASIC_CODEGENINTRINSICS_H
