; RUN: opt %loadNPMPolly -polly-optree-max-ops=1 '-passes=print<polly-optree>' -disable-output < %s | FileCheck %s -match-full-lines
; RUN: opt %loadNPMPolly -polly-optree-max-ops=1 -passes=polly-optree -disable-output -stats < %s 2>&1 | FileCheck %s -match-full-lines -check-prefix=STATS
; REQUIRES: asserts
;
; for (int j = 0; j < n; j += 1) {
; bodyA:
;   double val = B[j];
;
; bodyB:
;   A[j] = val;
; }
;
define void @func(i32 %n, ptr noalias nonnull %A, ptr noalias nonnull %B) {
entry:
  br label %for

for:
  %j = phi i32 [0, %entry], [%j.inc, %inc]
  %j.cmp = icmp slt i32 %j, %n
  br i1 %j.cmp, label %bodyA, label %exit

    bodyA:
      %B_idx = getelementptr inbounds double, ptr %B, i32 %j
      %val = load double, ptr %B_idx
      br label %bodyB

    bodyB:
      %A_idx = getelementptr inbounds double, ptr %A, i32 %j
      store double %val, ptr %A_idx
      br label %inc

inc:
  %j.inc = add nuw nsw i32 %j, 1
  br label %for

exit:
  br label %return

return:
  ret void
}


; CHECK: ForwardOpTree executed, but did not modify anything

; STATS-NOT: IMPLEMENTATION ERROR: Unhandled error state
; STATS: 1 polly-optree     - Analyses aborted because max_operations was reached
; STATS-NOT: IMPLEMENTATION ERROR: Unhandled error state
