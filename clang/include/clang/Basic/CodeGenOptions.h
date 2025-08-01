//===--- CodeGenOptions.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines the CodeGenOptions interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_CODEGENOPTIONS_H
#define LLVM_CLANG_BASIC_CODEGENOPTIONS_H

#include "clang/Basic/CFProtectionOptions.h"
#include "clang/Basic/PointerAuthOptions.h"
#include "clang/Basic/Sanitizers.h"
#include "clang/Basic/XRayInstr.h"
#include "llvm/ADT/FloatingPointMode.h"
#include "llvm/Frontend/Debug/Options.h"
#include "llvm/Frontend/Driver/CodeGenOptions.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Regex.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Transforms/Instrumentation/AddressSanitizerOptions.h"
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace llvm {
class PassBuilder;
}
namespace clang {

/// Bitfields of CodeGenOptions, split out from CodeGenOptions to ensure
/// that this large collection of bitfields is a trivial class type.
class CodeGenOptionsBase {
  friend class CompilerInvocation;
  friend class CompilerInvocationBase;

public:
  /// For ASTs produced with different option value, signifies their level of
  /// compatibility.
  enum class CompatibilityKind {
    /// Does affect the construction of the AST in a way that does prevent
    /// module interoperability.
    NotCompatible,
    /// Does affect the construction of the AST in a way that doesn't prevent
    /// interoperability (that is, the value can be different between an
    /// explicit module and the user of that module).
    Compatible,
    /// Does not affect the construction of the AST in any way (that is, the
    /// value can be different between an implicit module and the user of that
    /// module).
    Benign,
  };

  using CFBranchLabelSchemeKind = clang::CFBranchLabelSchemeKind;
  using ProfileInstrKind = llvm::driver::ProfileInstrKind;
  using AsanDetectStackUseAfterReturnMode =
      llvm::AsanDetectStackUseAfterReturnMode;
  using AsanDtorKind = llvm::AsanDtorKind;
  using VectorLibrary = llvm::driver::VectorLibrary;
  using ZeroCallUsedRegsKind = llvm::ZeroCallUsedRegs::ZeroCallUsedRegsKind;
  using WinX64EHUnwindV2Mode = llvm::WinX64EHUnwindV2Mode;

  using DebugCompressionType = llvm::DebugCompressionType;
  using EmitDwarfUnwindType = llvm::EmitDwarfUnwindType;
  using DebugTemplateNamesKind = llvm::codegenoptions::DebugTemplateNamesKind;
  using DebugInfoKind = llvm::codegenoptions::DebugInfoKind;
  using DebuggerKind = llvm::DebuggerKind;

#define CODEGENOPT(Name, Bits, Default, Compatibility) unsigned Name : Bits;
#define ENUM_CODEGENOPT(Name, Type, Bits, Default, Compatibility)
#include "clang/Basic/CodeGenOptions.def"

protected:
#define CODEGENOPT(Name, Bits, Default, Compatibility)
#define ENUM_CODEGENOPT(Name, Type, Bits, Default, Compatibility)              \
  unsigned Name : Bits;
#include "clang/Basic/CodeGenOptions.def"
};

/// CodeGenOptions - Track various options which control how the code
/// is optimized and passed to the backend.
class CodeGenOptions : public CodeGenOptionsBase {
public:
  enum InliningMethod {
    NormalInlining,     // Use the standard function inlining pass.
    OnlyHintInlining,   // Inline only (implicitly) hinted functions.
    OnlyAlwaysInlining  // Only run the always inlining pass.
  };

  enum ObjCDispatchMethodKind {
    Legacy = 0,
    NonLegacy = 1,
    Mixed = 2
  };

  enum TLSModel {
    GeneralDynamicTLSModel,
    LocalDynamicTLSModel,
    InitialExecTLSModel,
    LocalExecTLSModel
  };

  enum StructReturnConventionKind {
    SRCK_Default,  // No special option was passed.
    SRCK_OnStack,  // Small structs on the stack (-fpcc-struct-return).
    SRCK_InRegs    // Small structs in registers (-freg-struct-return).
  };

  enum EmbedBitcodeKind {
    Embed_Off,      // No embedded bitcode.
    Embed_All,      // Embed both bitcode and commandline in the output.
    Embed_Bitcode,  // Embed just the bitcode in the output.
    Embed_Marker    // Embed a marker as a placeholder for bitcode.
  };

  enum class ExtendVariableLivenessKind {
    None,
    This,
    All,
  };

  enum InlineAsmDialectKind {
    IAD_ATT,
    IAD_Intel,
  };

  enum DebugSrcHashKind {
    DSH_MD5,
    DSH_SHA1,
    DSH_SHA256,
    DSH_NONE,
  };

  // This field stores one of the allowed values for the option
  // -fbasic-block-sections=.  The allowed values with this option are:
  // {"all", "list=<file>", "none"}.
  //
  // "all" :        Generate basic block sections for all basic blocks.
  // "list=<file>": Generate basic block sections for a subset of basic blocks.
  //                The functions and the machine basic block ids are specified
  //                in the file.
  // "none":        Disable sections for basic blocks.
  std::string BBSections;

  // If set, override the default value of MCAsmInfo::BinutilsVersion. If
  // DisableIntegratedAS is specified, the assembly output will consider GNU as
  // support. "none" means that all ELF features can be used, regardless of
  // binutils support.
  std::string BinutilsVersion;

  enum class FramePointerKind {
    None,     // Omit all frame pointers.
    Reserved, // Maintain valid frame pointer chain.
    NonLeaf,  // Keep non-leaf frame pointers.
    All,      // Keep all frame pointers.
  };

  static StringRef getFramePointerKindName(FramePointerKind Kind) {
    switch (Kind) {
    case FramePointerKind::None:
      return "none";
    case FramePointerKind::Reserved:
      return "reserved";
    case FramePointerKind::NonLeaf:
      return "non-leaf";
    case FramePointerKind::All:
      return "all";
    }

    llvm_unreachable("invalid FramePointerKind");
  }

  /// Possible exception handling behavior.
  enum class ExceptionHandlingKind { None, SjLj, WinEH, DwarfCFI, Wasm };

  enum class SwiftAsyncFramePointerKind {
    Auto, // Choose Swift async extended frame info based on deployment target.
    Always, // Unconditionally emit Swift async extended frame info.
    Never,  // Don't emit Swift async extended frame info.
    Default = Always,
  };

  enum FiniteLoopsKind {
    Language, // Not specified, use language standard.
    Always,   // All loops are assumed to be finite.
    Never,    // No loop is assumed to be finite.
  };

  enum AssignmentTrackingOpts {
    Disabled,
    Enabled,
    Forced,
  };

  /// The code model to use (-mcmodel).
  std::string CodeModel;

  /// The code model-specific large data threshold to use
  /// (-mlarge-data-threshold).
  uint64_t LargeDataThreshold;

  /// The filename with path we use for coverage data files. The runtime
  /// allows further manipulation with the GCOV_PREFIX and GCOV_PREFIX_STRIP
  /// environment variables.
  std::string CoverageDataFile;

  /// The filename with path we use for coverage notes files.
  std::string CoverageNotesFile;

  /// Regexes separated by a semi-colon to filter the files to instrument.
  std::string ProfileFilterFiles;

  /// Regexes separated by a semi-colon to filter the files to not instrument.
  std::string ProfileExcludeFiles;

  /// The version string to put into coverage files.
  char CoverageVersion[4] = {'0', '0', '0', '0'};

  /// Enable additional debugging information.
  std::string DebugPass;

  /// The string to embed in debug information as the current working directory.
  std::string DebugCompilationDir;

  /// The string to embed in coverage mapping as the current working directory.
  std::string CoverageCompilationDir;

  /// The string to embed in the debug information for the compile unit, if
  /// non-empty.
  std::string DwarfDebugFlags;

  /// The string containing the commandline for the llvm.commandline metadata,
  /// if non-empty.
  std::string RecordCommandLine;

  llvm::SmallVector<std::pair<std::string, std::string>, 0> DebugPrefixMap;

  /// Prefix replacement map for source-based code coverage to remap source
  /// file paths in coverage mapping.
  llvm::SmallVector<std::pair<std::string, std::string>, 0> CoveragePrefixMap;

  /// The ABI to use for passing floating point arguments.
  std::string FloatABI;

  /// The file to use for dumping bug report by `Debugify` for original
  /// debug info.
  std::string DIBugsReportFilePath;

  /// The floating-point denormal mode to use.
  llvm::DenormalMode FPDenormalMode = llvm::DenormalMode::getIEEE();

  /// The floating-point denormal mode to use, for float.
  llvm::DenormalMode FP32DenormalMode = llvm::DenormalMode::getIEEE();

  /// The float precision limit to use, if non-empty.
  std::string LimitFloatPrecision;

  struct BitcodeFileToLink {
    /// The filename of the bitcode file to link in.
    std::string Filename;
    /// If true, we set attributes functions in the bitcode library according to
    /// our CodeGenOptions, much as we set attrs on functions that we generate
    /// ourselves.
    bool PropagateAttrs = false;
    /// If true, we use LLVM module internalizer.
    bool Internalize = false;
    /// Bitwise combination of llvm::Linker::Flags, passed to the LLVM linker.
    unsigned LinkFlags = 0;
  };

  /// The files specified here are linked in to the module before optimizations.
  std::vector<BitcodeFileToLink> LinkBitcodeFiles;

  /// The user provided name for the "main file", if non-empty. This is useful
  /// in situations where the input file name does not match the original input
  /// file, for example with -save-temps.
  std::string MainFileName;

  /// The name for the split debug info file used for the DW_AT_[GNU_]dwo_name
  /// attribute in the skeleton CU.
  std::string SplitDwarfFile;

  /// Output filename for the split debug info, not used in the skeleton CU.
  std::string SplitDwarfOutput;

  /// Output filename used in the COFF debug information.
  std::string ObjectFilenameForDebug;

  /// The name of the relocation model to use.
  llvm::Reloc::Model RelocationModel;

  /// If not an empty string, trap intrinsics are lowered to calls to this
  /// function instead of to trap instructions.
  std::string TrapFuncName;

  /// A list of dependent libraries.
  std::vector<std::string> DependentLibraries;

  /// A list of linker options to embed in the object file.
  std::vector<std::string> LinkerOptions;

  /// Name of the profile file to use as output for -fprofile-instr-generate,
  /// -fprofile-generate, and -fcs-profile-generate.
  std::string InstrProfileOutput;

  /// Name of the patchable function entry section with
  /// -fpatchable-function-entry.
  std::string PatchableFunctionEntrySection;

  /// Name of the profile file to use with -fprofile-sample-use.
  std::string SampleProfileFile;

  /// Name of the profile file to use as output for with -fmemory-profile.
  std::string MemoryProfileOutput;

  /// Name of the profile file to use as input for -fmemory-profile-use.
  std::string MemoryProfileUsePath;

  /// Name of the profile file to use as input for -fprofile-instr-use
  std::string ProfileInstrumentUsePath;

  /// Name of the profile remapping file to apply to the profile data supplied
  /// by -fprofile-sample-use or -fprofile-instr-use.
  std::string ProfileRemappingFile;

  /// Name of the function summary index file to use for ThinLTO function
  /// importing.
  std::string ThinLTOIndexFile;

  /// Name of a file that can optionally be written with minimized bitcode
  /// to be used as input for the ThinLTO thin link step, which only needs
  /// the summary and module symbol table (and not, e.g. any debug metadata).
  std::string ThinLinkBitcodeFile;

  /// Prefix to use for -save-temps output.
  std::string SaveTempsFilePrefix;

  /// Name of file passed with -fcuda-include-gpubinary option to forward to
  /// CUDA runtime back-end for incorporating them into host-side object file.
  std::string CudaGpuBinaryFileName;

  /// List of filenames passed in using the -fembed-offload-object option. These
  /// are offloading binaries containing device images and metadata.
  std::vector<std::string> OffloadObjects;

  /// The name of the file to which the backend should save YAML optimization
  /// records.
  std::string OptRecordFile;

  /// The regex that filters the passes that should be saved to the optimization
  /// records.
  std::string OptRecordPasses;

  /// The format used for serializing remarks (default: YAML)
  std::string OptRecordFormat;

  /// The name of the partition that symbols are assigned to, specified with
  /// -fsymbol-partition (see https://lld.llvm.org/Partitions.html).
  std::string SymbolPartition;

  /// If non-empty, allow the compiler to assume that the given source file
  /// identifier is unique at link time.
  std::string UniqueSourceFileIdentifier;

  enum RemarkKind {
    RK_Missing,            // Remark argument not present on the command line.
    RK_Enabled,            // Remark enabled via '-Rgroup'.
    RK_EnabledEverything,  // Remark enabled via '-Reverything'.
    RK_Disabled,           // Remark disabled via '-Rno-group'.
    RK_DisabledEverything, // Remark disabled via '-Rno-everything'.
    RK_WithPattern,        // Remark pattern specified via '-Rgroup=regexp'.
  };

  /// Optimization remark with an optional regular expression pattern.
  struct OptRemark {
    RemarkKind Kind = RK_Missing;
    std::string Pattern;
    std::shared_ptr<llvm::Regex> Regex;

    /// By default, optimization remark is missing.
    OptRemark() = default;

    /// Returns true iff the optimization remark holds a valid regular
    /// expression.
    bool hasValidPattern() const { return Regex != nullptr; }

    /// Matches the given string against the regex, if there is some.
    bool patternMatches(StringRef String) const {
      return hasValidPattern() && Regex->match(String);
    }
  };

  /// Selected optimizations for which we should enable optimization remarks.
  /// Transformation passes whose name matches the contained (optional) regular
  /// expression (and support this feature), will emit a diagnostic whenever
  /// they perform a transformation.
  OptRemark OptimizationRemark;

  /// Selected optimizations for which we should enable missed optimization
  /// remarks. Transformation passes whose name matches the contained (optional)
  /// regular expression (and support this feature), will emit a diagnostic
  /// whenever they tried but failed to perform a transformation.
  OptRemark OptimizationRemarkMissed;

  /// Selected optimizations for which we should enable optimization analyses.
  /// Transformation passes whose name matches the contained (optional) regular
  /// expression (and support this feature), will emit a diagnostic whenever
  /// they want to explain why they decided to apply or not apply a given
  /// transformation.
  OptRemark OptimizationRemarkAnalysis;

  /// Set of sanitizer checks that are non-fatal (i.e. execution should be
  /// continued when possible).
  SanitizerSet SanitizeRecover;

  /// Set of sanitizer checks that trap rather than diagnose.
  SanitizerSet SanitizeTrap;

  /// Set of sanitizer checks that can merge handlers (smaller code size at
  /// the expense of debuggability).
  SanitizerSet SanitizeMergeHandlers;

  /// Set of thresholds in a range [0.0, 1.0]: the top hottest code responsible
  /// for the given fraction of PGO counters will be excluded from sanitization
  /// (0.0 [default] to skip none, 1.0 to skip all).
  SanitizerMaskCutoffs SanitizeSkipHotCutoffs;

  /// Set of sanitizer checks, for which the instrumentation will be annotated
  /// with extra debug info.
  SanitizerSet SanitizeAnnotateDebugInfo;

  std::optional<double> AllowRuntimeCheckSkipHotCutoff;

  /// List of backend command-line options for -fembed-bitcode.
  std::vector<uint8_t> CmdArgs;

  /// A list of all -fno-builtin-* function names (e.g., memset).
  std::vector<std::string> NoBuiltinFuncs;

  std::vector<std::string> Reciprocals;

  /// Configuration for pointer-signing.
  PointerAuthOptions PointerAuth;

  /// The preferred width for auto-vectorization transforms. This is intended to
  /// override default transforms based on the width of the architected vector
  /// registers.
  std::string PreferVectorWidth;

  /// Set of XRay instrumentation kinds to emit.
  XRayInstrSet XRayInstrumentationBundle;

  std::vector<std::string> DefaultFunctionAttrs;

  /// List of dynamic shared object files to be loaded as pass plugins.
  std::vector<std::string> PassPlugins;

  /// List of pass builder callbacks.
  std::vector<std::function<void(llvm::PassBuilder &)>> PassBuilderCallbacks;

  /// List of global variables explicitly specified by the user as toc-data.
  std::vector<std::string> TocDataVarsUserSpecified;

  /// List of global variables that over-ride the toc-data default.
  std::vector<std::string> NoTocDataVars;

  /// Path to allowlist file specifying which objects
  /// (files, functions) should exclusively be instrumented
  /// by sanitizer coverage pass.
  std::vector<std::string> SanitizeCoverageAllowlistFiles;

  /// The guard style used for stack protector to get a initial value, this
  /// value usually be gotten from TLS or get from __stack_chk_guard, or some
  /// other styles we may implement in the future.
  std::string StackProtectorGuard;

  /// The TLS base register when StackProtectorGuard is "tls", or register used
  /// to store the stack canary for "sysreg".
  /// On x86 this can be "fs" or "gs".
  /// On AArch64 this can only be "sp_el0".
  std::string StackProtectorGuardReg;

  /// Specify a symbol to be the guard value.
  std::string StackProtectorGuardSymbol;

  /// Path to ignorelist file specifying which objects
  /// (files, functions) listed for instrumentation by sanitizer
  /// coverage pass should actually not be instrumented.
  std::vector<std::string> SanitizeCoverageIgnorelistFiles;

  /// Path to ignorelist file specifying which objects
  /// (files, functions) listed for instrumentation by sanitizer
  /// binary metadata pass should not be instrumented.
  std::vector<std::string> SanitizeMetadataIgnorelistFiles;

  /// Name of the stack usage file (i.e., .su file) if user passes
  /// -fstack-usage. If empty, it can be implied that -fstack-usage is not
  /// passed on the command line.
  std::string StackUsageOutput;

  /// Executable and command-line used to create a given CompilerInvocation.
  /// Most of the time this will be the full -cc1 command.
  const char *Argv0 = nullptr;
  std::vector<std::string> CommandLineArgs;

  /// The minimum hotness value a diagnostic needs in order to be included in
  /// optimization diagnostics.
  ///
  /// The threshold is an Optional value, which maps to one of the 3 states:
  /// 1. 0            => threshold disabled. All remarks will be printed.
  /// 2. positive int => manual threshold by user. Remarks with hotness exceed
  ///                    threshold will be printed.
  /// 3. None         => 'auto' threshold by user. The actual value is not
  ///                    available at command line, but will be synced with
  ///                    hotness threshold from profile summary during
  ///                    compilation.
  ///
  /// If threshold option is not specified, it is disabled by default.
  std::optional<uint64_t> DiagnosticsHotnessThreshold = 0;

  /// The maximum percentage profiling weights can deviate from the expected
  /// values in order to be included in misexpect diagnostics.
  std::optional<uint32_t> DiagnosticsMisExpectTolerance = 0;

  /// The name of a file to use with \c .secure_log_unique directives.
  std::string AsSecureLogFile;

  /// A list of functions that are replacable by the loader.
  std::vector<std::string> LoaderReplaceableFunctionNames;
  /// The name of a file that contains functions which will be compiled for
  /// hotpatching. See -fms-secure-hotpatch-functions-file.
  std::string MSSecureHotPatchFunctionsFile;

  /// A list of functions which will be compiled for hotpatching.
  /// See -fms-secure-hotpatch-functions-list.
  std::vector<std::string> MSSecureHotPatchFunctionsList;

public:
  // Define accessors/mutators for code generation options of enumeration type.
#define CODEGENOPT(Name, Bits, Default, Compatibility)
#define ENUM_CODEGENOPT(Name, Type, Bits, Default, Compatibility)              \
  Type get##Name() const { return static_cast<Type>(Name); }                   \
  void set##Name(Type Value) { Name = static_cast<unsigned>(Value); }
#include "clang/Basic/CodeGenOptions.def"

  CodeGenOptions();

  const std::vector<std::string> &getNoBuiltinFuncs() const {
    return NoBuiltinFuncs;
  }

  bool hasSjLjExceptions() const {
    return getExceptionHandling() == ExceptionHandlingKind::SjLj;
  }

  bool hasSEHExceptions() const {
    return getExceptionHandling() == ExceptionHandlingKind::WinEH;
  }

  bool hasDWARFExceptions() const {
    return getExceptionHandling() == ExceptionHandlingKind::DwarfCFI;
  }

  bool hasWasmExceptions() const {
    return getExceptionHandling() == ExceptionHandlingKind::Wasm;
  }

  /// Check if Clang profile instrumenation is on.
  bool hasProfileClangInstr() const {
    return getProfileInstr() ==
           llvm::driver::ProfileInstrKind::ProfileClangInstr;
  }

  /// Check if IR level profile instrumentation is on.
  bool hasProfileIRInstr() const {
    return getProfileInstr() == llvm::driver::ProfileInstrKind::ProfileIRInstr;
  }

  /// Check if CS IR level profile instrumentation is on.
  bool hasProfileCSIRInstr() const {
    return getProfileInstr() ==
           llvm::driver::ProfileInstrKind::ProfileCSIRInstr;
  }

  /// Check if any form of instrumentation is on.
  bool hasProfileInstr() const {
    return getProfileInstr() != llvm::driver::ProfileInstrKind::ProfileNone;
  }

  /// Check if Clang profile use is on.
  bool hasProfileClangUse() const {
    return getProfileUse() == llvm::driver::ProfileInstrKind::ProfileClangInstr;
  }

  /// Check if IR level profile use is on.
  bool hasProfileIRUse() const {
    return getProfileUse() == llvm::driver::ProfileInstrKind::ProfileIRInstr ||
           getProfileUse() == llvm::driver::ProfileInstrKind::ProfileCSIRInstr;
  }

  /// Check if CSIR profile use is on.
  bool hasProfileCSIRUse() const {
    return getProfileUse() == llvm::driver::ProfileInstrKind::ProfileCSIRInstr;
  }

  /// Check if type and variable info should be emitted.
  bool hasReducedDebugInfo() const {
    return getDebugInfo() >= llvm::codegenoptions::DebugInfoConstructor;
  }

  /// Check if maybe unused type info should be emitted.
  bool hasMaybeUnusedDebugInfo() const {
    return getDebugInfo() >= llvm::codegenoptions::UnusedTypeInfo;
  }

  // Check if any one of SanitizeCoverage* is enabled.
  bool hasSanitizeCoverage() const {
    return SanitizeCoverageType || SanitizeCoverageIndirectCalls ||
           SanitizeCoverageTraceCmp || SanitizeCoverageTraceLoads ||
           SanitizeCoverageTraceStores || SanitizeCoverageControlFlow;
  }

  // Check if any one of SanitizeBinaryMetadata* is enabled.
  bool hasSanitizeBinaryMetadata() const {
    return SanitizeBinaryMetadataCovered || SanitizeBinaryMetadataAtomics ||
           SanitizeBinaryMetadataUAR;
  }

  /// Reset all of the options that are not considered when building a
  /// module.
  void resetNonModularOptions(StringRef ModuleFormat);

  // Is the given function name one of the functions that can be replaced by the
  // loader?
  bool isLoaderReplaceableFunctionName(StringRef FuncName) const {
    return llvm::is_contained(LoaderReplaceableFunctionNames, FuncName);
  }
};

}  // end namespace clang

#endif
