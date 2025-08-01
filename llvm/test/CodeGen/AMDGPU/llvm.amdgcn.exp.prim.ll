; RUN: llc -mtriple=amdgcn -mcpu=tonga < %s | FileCheck -strict-whitespace -check-prefix=GCN -check-prefix=NOPRIM %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx1010 < %s | FileCheck -strict-whitespace -check-prefix=GCN -check-prefix=PRIM %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx1100 < %s | FileCheck -strict-whitespace -check-prefix=GCN -check-prefix=PRIM %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx1200 < %s | FileCheck -strict-whitespace -check-prefix=GCN -check-prefix=PRIM %s

declare void @llvm.amdgcn.exp.i32(i32, i32, i32, i32, i32, i32, i1, i1) #1

; GCN-LABEL: {{^}}test_export_prim_i32:
; NOPRIM: exp invalid_target_20 v0, off, off, off done{{$}}
; PRIM: {{exp|export}} prim v0, off, off, off done{{$}}
define amdgpu_gs void @test_export_prim_i32(i32 inreg %a) #0 {
  call void @llvm.amdgcn.exp.i32(i32 20, i32 1, i32 %a, i32 poison, i32 poison, i32 poison, i1 true, i1 false)
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind inaccessiblememonly }
