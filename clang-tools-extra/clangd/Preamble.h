//===--- Preamble.h - Reusing expensive parts of the AST ---------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The vast majority of code in a typical translation unit is in the headers
// included at the top of the file.
//
// The preamble optimization says that we can parse this code once, and reuse
// the result multiple times. The preamble is invalidated by changes to the
// code in the preamble region, to the compile command, or to files on disk.
//
// This is the most important optimization in clangd: it allows operations like
// code-completion to have sub-second latency. It is supported by the
// PrecompiledPreamble functionality in clang, which wraps the techniques used
// by PCH files, modules etc into a convenient interface.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_PREAMBLE_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_PREAMBLE_H

#include "CollectMacros.h"
#include "Compiler.h"
#include "Diagnostics.h"
#include "FS.h"
#include "Headers.h"
#include "ModulesBuilder.h"

#include "clang-include-cleaner/Record.h"
#include "support/Path.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/PrecompiledPreamble.h"
#include "clang/Lex/Lexer.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace clang {
namespace clangd {

/// The captured AST context.
/// Keeps necessary structs for an ASTContext and Preprocessor alive.
/// This enables consuming them after context that produced the AST is gone.
/// (e.g. indexing a preamble ast on a separate thread). ASTContext stored
/// inside is still not thread-safe.

struct CapturedASTCtx {
public:
  CapturedASTCtx(CompilerInstance &Clang)
      : Invocation(Clang.getInvocationPtr()),
        Diagnostics(Clang.getDiagnosticsPtr()), Target(Clang.getTargetPtr()),
        AuxTarget(Clang.getAuxTarget()), FileMgr(Clang.getFileManagerPtr()),
        SourceMgr(Clang.getSourceManagerPtr()), PP(Clang.getPreprocessorPtr()),
        Context(Clang.getASTContextPtr()) {}

  CapturedASTCtx(const CapturedASTCtx &) = delete;
  CapturedASTCtx &operator=(const CapturedASTCtx &) = delete;
  CapturedASTCtx(CapturedASTCtx &&) = default;
  CapturedASTCtx &operator=(CapturedASTCtx &&) = default;

  ASTContext &getASTContext() { return *Context; }
  Preprocessor &getPreprocessor() { return *PP; }
  CompilerInvocation &getCompilerInvocation() { return *Invocation; }
  FileManager &getFileManager() { return *FileMgr; }
  void setStatCache(std::shared_ptr<PreambleFileStatusCache> StatCache) {
    this->StatCache = StatCache;
  }

private:
  std::shared_ptr<CompilerInvocation> Invocation;
  IntrusiveRefCntPtr<DiagnosticsEngine> Diagnostics;
  IntrusiveRefCntPtr<TargetInfo> Target;
  IntrusiveRefCntPtr<TargetInfo> AuxTarget;
  IntrusiveRefCntPtr<FileManager> FileMgr;
  IntrusiveRefCntPtr<SourceManager> SourceMgr;
  std::shared_ptr<Preprocessor> PP;
  IntrusiveRefCntPtr<ASTContext> Context;
  std::shared_ptr<PreambleFileStatusCache> StatCache;
};

/// The parsed preamble and associated data.
///
/// As we must avoid re-parsing the preamble, any information that can only
/// be obtained during parsing must be eagerly captured and stored here.
struct PreambleData {
  PreambleData(PrecompiledPreamble Preamble) : Preamble(std::move(Preamble)) {}

  // Version of the ParseInputs this preamble was built from.
  std::string Version;
  tooling::CompileCommand CompileCommand;
  // Target options used when building the preamble. Changes in target can cause
  // crashes when deserializing preamble, this enables consumers to use the
  // same target (without reparsing CompileCommand).
  std::unique_ptr<TargetOptions> TargetOpts = nullptr;
  PrecompiledPreamble Preamble;
  std::vector<Diag> Diags;
  // Processes like code completions and go-to-definitions will need #include
  // information, and their compile action skips preamble range.
  IncludeStructure Includes;
  // Captures #include-mapping information in #included headers.
  std::shared_ptr<const include_cleaner::PragmaIncludes> Pragmas;
  // Information about required module files for this preamble.
  std::unique_ptr<PrerequisiteModules> RequiredModules;
  // Macros defined in the preamble section of the main file.
  // Users care about headers vs main-file, not preamble vs non-preamble.
  // These should be treated as main-file entities e.g. for code completion.
  MainFileMacros Macros;
  // Pragma marks defined in the preamble section of the main file.
  std::vector<PragmaMark> Marks;
  // Cache of FS operations performed when building the preamble.
  // When reusing a preamble, this cache can be consumed to save IO.
  std::shared_ptr<PreambleFileStatusCache> StatCache;
  // Whether there was a (possibly-incomplete) include-guard on the main file.
  // We need to propagate this information "by hand" to subsequent parses.
  bool MainIsIncludeGuarded = false;
};

using PreambleParsedCallback =
    std::function<void(CapturedASTCtx ASTCtx,
                       std::shared_ptr<const include_cleaner::PragmaIncludes>)>;

/// Timings and statistics from the premble build. Unlike PreambleData, these
/// do not need to be stored for later, but can be useful for logging, metrics,
/// etc.
struct PreambleBuildStats {
  /// Total wall time it took to build preamble, in seconds.
  double TotalBuildTime;
  /// Time spent in filesystem operations during the build, in seconds.
  double FileSystemTime;

  /// Estimate of the memory used while building the preamble.
  /// This memory has been released when buildPreamble returns.
  /// For example, this includes the size of the in-memory AST (ASTContext).
  size_t BuildSize;
  /// The serialized size of the preamble.
  /// This storage is needed while the preamble is used (but may be on disk).
  size_t SerializedSize;
};

/// Build a preamble for the new inputs unless an old one can be reused.
/// If \p PreambleCallback is set, it will be run on top of the AST while
/// building the preamble.
/// If Stats is not non-null, build statistics will be exported there.
std::shared_ptr<const PreambleData>
buildPreamble(PathRef FileName, CompilerInvocation CI,
              const ParseInputs &Inputs, bool StoreInMemory,
              PreambleParsedCallback PreambleCallback,
              PreambleBuildStats *Stats = nullptr);

/// Returns true if \p Preamble is reusable for \p Inputs. Note that it will
/// return true when some missing headers are now available.
/// FIXME: Should return more information about the delta between \p Preamble
/// and \p Inputs, e.g. new headers.
bool isPreambleCompatible(const PreambleData &Preamble,
                          const ParseInputs &Inputs, PathRef FileName,
                          const CompilerInvocation &CI);

/// Stores information required to parse a TU using a (possibly stale) Baseline
/// preamble. Later on this information can be injected into the main file by
/// updating compiler invocation with \c apply. This injected section
/// approximately reflects additions to the preamble in Modified contents, e.g.
/// new include directives.
class PreamblePatch {
public:
  enum class PatchType { MacroDirectives, All };
  /// \p Preamble is used verbatim.
  static PreamblePatch unmodified(const PreambleData &Preamble);
  /// Builds a patch that contains new PP directives introduced to the preamble
  /// section of \p Modified compared to \p Baseline.
  /// FIXME: This only handles include directives, we should at least handle
  /// define/undef.
  static PreamblePatch createFullPatch(llvm::StringRef FileName,
                                       const ParseInputs &Modified,
                                       const PreambleData &Baseline);
  static PreamblePatch createMacroPatch(llvm::StringRef FileName,
                                        const ParseInputs &Modified,
                                        const PreambleData &Baseline);
  /// Returns the FileEntry for the preamble patch of MainFilePath in SM, if
  /// any.
  static OptionalFileEntryRef getPatchEntry(llvm::StringRef MainFilePath,
                                            const SourceManager &SM);

  /// Adjusts CI (which compiles the modified inputs) to be used with the
  /// baseline preamble. This is done by inserting an artificial include to the
  /// \p CI that contains new directives calculated in create.
  void apply(CompilerInvocation &CI) const;

  /// Returns #include directives from the \c Modified preamble that were
  /// resolved using the \c Baseline preamble. This covers the new locations of
  /// inclusions that were moved around, but not inclusions of new files. Those
  /// will be recorded when parsing the main file: the includes in the injected
  /// section will be resolved back to their spelled positions in the main file
  /// using the presumed-location mechanism.
  std::vector<Inclusion> preambleIncludes() const;

  /// Returns preamble bounds for the Modified.
  PreambleBounds modifiedBounds() const { return ModifiedBounds; }

  /// Returns textual patch contents.
  llvm::StringRef text() const { return PatchContents; }

  /// Returns diag locations for Modified contents.
  llvm::ArrayRef<Diag> patchedDiags() const { return PatchedDiags; }

  static constexpr llvm::StringLiteral HeaderName = "__preamble_patch__.h";

  llvm::ArrayRef<PragmaMark> marks() const;
  const MainFileMacros &mainFileMacros() const;

private:
  static PreamblePatch create(llvm::StringRef FileName,
                              const ParseInputs &Modified,
                              const PreambleData &Baseline,
                              PatchType PatchType);

  PreamblePatch() = default;
  std::string PatchContents;
  std::string PatchFileName;
  // Includes that are present in both Baseline and Modified. Used for
  // patching includes of baseline preamble.
  std::vector<Inclusion> PreambleIncludes;
  // Diags that were attached to a line preserved in Modified contents.
  std::vector<Diag> PatchedDiags;
  PreambleBounds ModifiedBounds = {0, false};
  const PreambleData *Baseline = nullptr;
  std::vector<PragmaMark> PatchedMarks;
  MainFileMacros PatchedMacros;
};

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_PREAMBLE_H
