!-----------------------------------------------------------------------------------------
module nn4da3d
  implicit none

!-----------------------------------------------------------------------------------------
  type anl_weight_type
    real,    dimension(3,3,3) :: wgt
    real,    dimension(3,3,3) :: grd
    integer, dimension(3,3,3) :: cnt
    real :: bias, bias_grd
    integer :: bias_cnt, ibase, jbase, kbase
  end type anl_weight_type

!-----------------------------------------------------------------------------------------
  type obs_weight_type
    real, dimension(2, 2, 2) :: wgt
    integer :: ibase, jbase, kbase
  end type obs_weight_type

!-----------------------------------------------------------------------------------------
  integer :: nlon, nlat, nprs, nobs
  
  type(anl_weight_type), dimension(:,:,:), allocatable :: wgt
  type(obs_weight_type), dimension(:),   allocatable :: obs_wgt

  real, dimension(:),   allocatable :: lon, lat, ak, bk
  real, dimension(:),   allocatable :: obslon, obslat, obsprs, &
                                       obsval, obsanl, obserr

  real, dimension(:,:),   allocatable :: ps
  real, dimension(:,:,:), allocatable :: xa, xb

  real :: offset, scalefactor
  real :: deltlon, deltlat

end module nn4da3d

