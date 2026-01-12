!RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=60 %s -o - | FileCheck %s

! Test lowering of groupprivate directive to omp.groupprivate operations
! Groupprivate variables are NOT implicitly mapped - they use fir.address_of
! to access the global directly. Type information is obtained from the global
! variable in LLVM IR translation to create LDS storage.

module m
  implicit none
  integer, save :: x
  !$omp groupprivate(x)
end module

! CHECK: fir.global @_QMmEx : i32

! CHECK-LABEL: func.func @_QPtest_groupprivate
subroutine test_groupprivate()
  use m

  !$omp target
    !$omp teams
      x = 10
    !$omp end teams
  !$omp end target
end subroutine

! The groupprivate transformation happens when entering the teams region.
! Groupprivate vars use fir.address_of to get the global address inside teams.
! CHECK: omp.target
! CHECK:   omp.teams
! CHECK:     fir.address_of(@_QMmEx)
! CHECK:     omp.groupprivate

! Test groupprivate with multiple variables
module m2
  implicit none
  integer, save :: a, b
  !$omp groupprivate(a, b)
end module

! CHECK: fir.global @_QMm2Ea : i32
! CHECK: fir.global @_QMm2Eb : i32

! CHECK-LABEL: func.func @_QPtest_multiple_groupprivate
subroutine test_multiple_groupprivate()
  use m2

  !$omp target
    !$omp teams
      a = 1
      b = 2
    !$omp end teams
  !$omp end target
end subroutine

! CHECK: omp.target
! CHECK:   omp.teams
! CHECK:     fir.address_of(@_QMm2Ea)
! CHECK:     omp.groupprivate
! CHECK:     fir.address_of(@_QMm2Eb)
! CHECK:     omp.groupprivate

! Test groupprivate in teams outside target (host execution)
module m3
  implicit none
  integer, save :: y
  !$omp groupprivate(y)
end module

! CHECK: fir.global @_QMm3Ey : i32

! CHECK-LABEL: func.func @_QPtest_groupprivate_host
subroutine test_groupprivate_host()
  use m3

  !$omp teams
    y = 100
  !$omp end teams
end subroutine

! On host (no target), groupprivate also uses fir.address_of inside teams
! CHECK: omp.teams
! CHECK:   fir.address_of(@_QMm3Ey)
! CHECK:   omp.groupprivate
