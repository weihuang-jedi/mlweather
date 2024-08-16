!-----------------------------------------------------------------------------------------
subroutine initialize(mlon, mlat, mprs, mobs, loni, lati, aki, bki, psi, xbi, &
                      obsloni, obslati, obsprsi, obsvali, iflag)
  use nn4da3d
  implicit none

  integer, intent(in) :: mlon, mlat, mprs, mobs

  real, dimension(mlon), intent(in) :: loni
  real, dimension(mlat), intent(in) :: lati
  real, dimension(mprs+1), intent(in) :: aki, bki
  real, dimension(mlat, mlon), intent(in) :: psi
  real, dimension(mprs, mlat, mlon), intent(in) :: xbi
  real, dimension(mobs), intent(in) :: obsloni, obslati, obsprsi, obsvali

  integer, intent(out) :: iflag

  integer :: i, j, k, n

 !print *, 'Enter initialize'
 !print *, 'size(loni) = ', size(loni)
 !print *, 'size(lati) = ', size(lati)
 !print *, 'size(pri) = ', size(psi)
 !print *, 'size(xbi) = ', size(xbi)
 !print *, 'size(xbi, dim=1) = ', size(xbi, dim=1)
 !print *, 'size(xbi, dim=2) = ', size(xbi, dim=2)
 !print *, 'size(xbi, dim=3) = ', size(xbi, dim=3)
 !print *, 'size(obsloni) = ', size(obsloni)
 !print *, 'size(obslati) = ', size(obslati)
 !print *, 'size(obsprsi) = ', size(obsprsi)
 !print *, 'size(obsvali) = ', size(obsvali)

  offset = 0.0
  scalefactor = 1.0

  nlon = mlon
  nlat = mlat
  nprs = mprs
  nobs = mobs
 
  if(.not. allocated(wgt)) allocate(wgt(-1:nlon+2, -1:nlat+2, -1:nprs+2))
  if(.not. allocated(obs_wgt)) allocate(obs_wgt(nobs))

  if(.not. allocated(lon)) allocate(lon(-1:nlon+2))
  if(.not. allocated(lat)) allocate(lat(-1:nlat+2))
  if(.not. allocated(ak)) allocate(ak(nprs+1))
  if(.not. allocated(bk)) allocate(bk(nprs+1))
  if(.not. allocated(obslon)) allocate(obslon(nobs))
  if(.not. allocated(obslat)) allocate(obslat(nobs))
  if(.not. allocated(obsprs)) allocate(obsprs(nobs))
  if(.not. allocated(obsval)) allocate(obsval(nobs))
  if(.not. allocated(obsanl)) allocate(obsanl(nobs))
  if(.not. allocated(obserr)) allocate(obserr(nobs))
  if(.not. allocated(ps)) allocate(ps(-1:nlon+2, -1:nlat+2))
  if(.not. allocated(xa)) allocate(xa(-1:nlon+2, -1:nlat+2, -1:nprs+2))
  if(.not. allocated(xb)) allocate(xb(-1:nlon+2, -1:nlat+2, -1:nprs+2))

 !print *, 'size(wgt) = ', size(wgt)
 !print *, 'size(obs_wgt) = ', size(obs_wgt)
 !print *, 'size(lon) = ', size(lon)
 !print *, 'size(lat) = ', size(lat)
 !print *, 'size(obslon) = ', size(obslon)
 !print *, 'size(obslat) = ', size(obslat)
 !print *, 'size(obsanl) = ', size(obsanl)
 !print *, 'size(xa) = ', size(xa)
 !print *, 'size(xa, dim=1) = ', size(xa, dim=1)
 !print *, 'size(xa, dim=2) = ', size(xa, dim=2)
 !print *, 'size(xa, dim=3) = ', size(xa, dim=3)
 !print *, 'size(xb) = ', size(xb)
 !print *, 'size(xb, dim=1) = ', size(xb, dim=1)
 !print *, 'size(xb, dim=2) = ', size(xb, dim=2)
 !print *, 'size(xb, dim=3) = ', size(xb, dim=3)

  do i=1,nlon
    lon(i) = loni(i)
   !print *, 'lon(', i, ')=', lon(i)
  end do

  do j=1,nlat
    lat(j) = lati(j)
   !print *, 'lat(', j, ')=', lat(j)
    do k=1,nprs
    do i=1,nlon
      xb(i,j,k) = xbi(k,j,i)
     !print *, 'i,j,k,xb(i,j,k) =', i,j,k,xb(i,j,k)
    end do
    end do
  end do

  do j=1,nlat
    do i=1,nlon
      ps(i,j) = psi(j,i)
    end do
  end do

  deltlon = lon(2) - lon(1)
  deltlat = lat(2) - lat(1)
 !print *, 'deltlon, deltlat = ', deltlon, deltlat

  lon( 0) = lon(1) - deltlon
  lon(-1) = lon(0) - deltlon
  lon(nlon+1) = lon(nlon) + deltlon
  lon(nlon+2) = lon(nlon+1) + deltlon

  lat( 0) = lat(1) - deltlat
  lat(-1) = lat(0) - deltlat
  lat(nlat+1) = lat(nlat) + deltlat
  lat(nlat+2) = lat(nlat+1) + deltlat

  do k=1,nprs+1
    ak(k) = aki(k)
    bk(k) = bki(k)
  end do

  call extend_boundary(nlon, nlat, nprs, xb)
  call extend_boundary_2d(nlon, nlat, ps)

  do k=-1,nprs+2
  do j=-1,nlat+2
  do i=-1,nlon+2
    call random_number(wgt(i,j,k)%wgt)
    wgt(i,j,k)%wgt = 0.001*(wgt(i,j,k)%wgt - 0.5)
    wgt(i,j,k)%wgt(2,2,2) = 1.0 + wgt(i,j,k)%wgt(2,2,2)
    wgt(i,j,k)%grd = 0.0
    wgt(i,j,k)%cnt = 0
    call random_number(wgt(i,j,k)%bias)
    wgt(i,j,k)%bias = 0.001*(wgt(i,j,k)%bias - 0.5)
    wgt(i,j,k)%bias_grd = 0.0
    wgt(i,j,k)%bias_cnt = 0
    wgt(i,j,k)%ibase = i - 2
    wgt(i,j,k)%jbase = j - 2
    wgt(i,j,k)%kbase = k - 2

   !print *, 'i, j, k, wgt(i,j,k)%bias = ', i, j, k, wgt(i,j,k)%bias
   !print *, 'i, j, k, wgt(i,j,k)%wgt = ', i, j, k, wgt(i,j,k)%wgt
  end do
  end do
  end do

  do n=1, nobs
    obslon(n) = obsloni(n)
    obslat(n) = obslati(n)
    obsprs(n) = obsprsi(n)
    obsval(n) = obsvali(n)

   !print *, 'n, obslon(n), obslat(n), obsval(n)=', n, obslon(n), obslat(n), obsval(n)
  end do

  call interp(iflag)
     
 !print *, 'Leave initialize'

  return
end subroutine initialize

!-----------------------------------------------------------------------------------------
subroutine reset_xb(mlon, mlat, mprs, xbi, iflag)
  use nn4da3d
  implicit none

  integer, intent(in) :: mlon, mlat, mprs
  real, dimension(mprs, mlat, mlon), intent(in) :: xbi
  integer, intent(out) :: iflag

  integer :: i, j, k

  do k=1,mprs
  do j=1,mlat
  do i=1,mlon
    xb(i,j,k) = xbi(k,j,i)
  end do
  end do
  end do

  call extend_boundary(nlon, nlat, nprs, xb)
  call reset_wgt(iflag)

  return
end subroutine reset_xb

!-----------------------------------------------------------------------------------------
subroutine extend_boundary(nlon, nlat, nprs, xc)
  implicit none

  integer, intent(in) :: nlon, nlat, nprs
  real, dimension(-1:nlon+2, -1:nlat+2, -1:nprs+2), intent(inout) :: xc

  integer i, j, k, im, iop

  iop = nlon/2
  
  do k=1,nprs
  do j=1,nlat
    xc(-1,j,k) = xc(nlon-1,j,k)
    xc(0,j,k) = xc(nlon,j,k)
    xc(nlon+1,j,k) = xc(1,j,k)
    xc(nlon+2,j,k) = xc(2,j,k)
  end do

  do i=-1,nlon+2
    im = i + iop
    if(im > nlon) im = im - nlon
    xc(i,0,k) = xc(im,1,k)
    xc(i,-1,k) = xc(im,2,k)
    xc(i,nlat+1,k) = xc(im,nlat-1,k)
    xc(i,nlat+2,k) = xc(im,nlat-2,k)
  end do
  end do

  do j=-1,nlat+2
  do i=-1,nlon+2
    xc(i,j,0) = xc(i,2,k)
    xc(i,j,-1) = xc(i,3,k)
    xc(i,j,nprs+1) = xc(i,nprs-1,k)
    xc(i,j,nprs+2) = xc(i,nprs-2,k)
  end do
  end do

  return
end subroutine extend_boundary

!-----------------------------------------------------------------------------------------
subroutine extend_boundary_2d(nlon, nlat, xc)
  implicit none

  integer, intent(in) :: nlon, nlat
  real, dimension(-1:nlon+2, -1:nlat+2), intent(inout) :: xc

  integer i, j, im, iop

  iop = nlon/2
  
  do j=1,nlat
    xc(-1,j) = xc(nlon-1,j)
    xc(0,j) = xc(nlon,j)
    xc(nlon+1,j) = xc(1,j)
    xc(nlon+2,j) = xc(2,j)
  end do

  do i=-1,nlon+2
    im = i + iop
    if(im > nlon) im = im - nlon
    xc(i,0) = xc(im,1)
    xc(i,-1) = xc(im,2)
    xc(i,nlat+1) = xc(im,nlat-1)
    xc(i,nlat+2) = xc(im,nlat-2)
  end do

  return
end subroutine extend_boundary_2d

!-----------------------------------------------------------------------------------------
subroutine finalize(iflag)
  use nn4da3d

  implicit none

  integer, intent(out) :: iflag

  if(allocated(wgt)) deallocate(wgt)
  if(allocated(obs_wgt)) deallocate(obs_wgt)

  if(allocated(lon)) deallocate(lon)
  if(allocated(lat)) deallocate(lat)
  if(allocated(ak)) deallocate(ak)
  if(allocated(bk)) deallocate(bk)

  if(allocated(obslon)) deallocate(obslon)
  if(allocated(obslat)) deallocate(obslat)
  if(allocated(obsprs)) deallocate(obsprs)
  if(allocated(obsval)) deallocate(obsval)
  if(allocated(obsanl)) deallocate(obsanl)
  if(allocated(obserr)) deallocate(obserr)

  if(allocated(ps)) deallocate(ps)
  if(allocated(xa)) deallocate(xa)
  if(allocated(xb)) deallocate(xb)

  iflag = 0
    
  return
end subroutine finalize

!-----------------------------------------------------------------------------------------
subroutine interp(iflag)
  use nn4da3d
  implicit none
  integer, intent(out) :: iflag

  integer :: i, j, n
  real, dimension(2,2) :: wgt2d
  real :: dx, dy, dz

 !print *, 'deltlon, deltlat = ', deltlon, deltlat

 !interpolate from xb to yb
  do n = 1, nobs
    i = int((obslon(n))/deltlon)
    j = int((obslat(n)+90.0)/deltlat)

    obs_wgt(n)%ibase = i
    obs_wgt(n)%jbase = j

    dx = (obslon(n) - lon(i+1))/deltlon
    dy = (obslat(n) - lat(j+1))/deltlat

   !print *, 'n, obslon(n), obslat(n), i, j = ', n, obslon(n), obslat(n), i, j
   !print *, 'n, obslon(n), lon(i+1), obslat(n), lat(j+1), dx, dy = ', &
   !          n, obslon(n), lon(i+1), obslat(n), lat(j+1), dx, dy

    wgt2d(1,1) = (1.0-dx)*(1.0-dy)
    wgt2d(1,2) = (1.0-dx)*dy
    wgt2d(2,1) = dx*(1.0-dy)
    wgt2d(2,2) = dx*dy

    call vertical_interp(i, j, wgt2d, obsprs(n), dz, obs_wgt(n)%kbase)

    obs_wgt(n)%wgt(1,1,1) = wgt2d(1,1)*(1.0-dz)
    obs_wgt(n)%wgt(1,2,1) = wgt2d(1,2)*(1.0-dz)
    obs_wgt(n)%wgt(2,1,1) = wgt2d(2,1)*(1.0-dz)
    obs_wgt(n)%wgt(2,2,1) = wgt2d(2,2)*(1.0-dz)
    obs_wgt(n)%wgt(1,1,2) = wgt2d(1,1)*dz
    obs_wgt(n)%wgt(1,2,2) = wgt2d(1,2)*dz
    obs_wgt(n)%wgt(2,1,2) = wgt2d(2,1)*dz
    obs_wgt(n)%wgt(2,2,2) = wgt2d(2,2)*dz

   !print *, 'Obs ', n, ', wgt: ', obs_wgt(n)%wgt, ', sum(obs_wgt(n)%wgt) = ', sum(obs_wgt(n)%wgt)
   !print *, 'Obs ', n, ', obs_wgt(n)%ibase, obs_wgt(n)%jbase, obs_wgt(n)%kbase=', &
   !         obs_wgt(n)%ibase, obs_wgt(n)%jbase, obs_wgt(n)%kbase
  end do

  iflag = 0
        
  return
end subroutine interp

!-----------------------------------------------------------------------------------------
subroutine vertical_interp(i, j, wgt2d, op, dz, kb)
  use nn4da3d
  implicit none
  integer,              intent(in)  :: i, j
  real, dimension(2,2), intent(in)  :: wgt2d
  real,                 intent(in)  :: op
  real,                 intent(out) :: dz
  integer,              intent(out) :: kb

  integer :: m, n, k
  real, dimension(nprs+1) :: pfull
  real, dimension(nprs)   :: phalf
  real :: psf

  psf = 0.0
  do n=1,2
  do m=1,2
    psf = psf + wgt2d(m,n)*ps(i+m,j+n)
  end do
  end do

  do k=1, nprs+1
    pfull(k) = ak(k) + bk(k)*psf
  end do

  do k=1, nprs
   phalf(k) = 0.5*(pfull(k) + pfull(k+1))
  end do

  if(op <= phalf(1)) then
    kb = 0
    dz = 0.0
  else if(op >= phalf(nprs)) then
    kb = nprs - 2
    dz = 1.0
  else
    do k=2,nprs
      if(op <= phalf(k)) then
        kb = k-2
        dz = (op - phalf(k-1))/(phalf(k) - phalf(k-1))
        exit
      end if
    end do
  end if

 !print *, 'i,j,ps(i+1,j+1),psf,kb,dz,phalf(kb+1)=', i,j,ps(i+1,j+1),psf,kb,dz,phalf(kb+1)

  return
end subroutine vertical_interp

!-----------------------------------------------------------------------------------------
subroutine update(iflag)
  use nn4da3d
  implicit none
  integer, intent(out) :: iflag

  integer :: i, j, k, l, m, n

  do k=-1,nprs+2
  do j=-1,nlat+2
  do i=-1,nlon+2
    xa(i,j,k) = wgt(i,j,k)%bias
    do n = 1, 3
    do m = 1, 3
    do l = 1, 3
      xa(i,j,k) = xa(i,j,k) + wgt(i,j,k)%wgt(l,m,n)*xb(i+l-2,j+m-2,k+n-2)
    end do
    end do
    end do
  end do
  end do
  end do

  call extend_boundary(nlon, nlat, nprs, xa)

  iflag = 0

  return
end subroutine update

!-----------------------------------------------------------------------------------------
subroutine forward(cost)
  use nn4da3d
  implicit none
  real, intent(out) :: cost

  integer :: i, j, k, l, m, n, no

  call update(no)

  cost = 0.0
  do no=1,nobs
    i = obs_wgt(no)%ibase
    j = obs_wgt(no)%jbase
    k = obs_wgt(no)%kbase
    obsanl(no) = 0.0
    do n=1,2
    do m=1,2
    do l=1,2
      obsanl(no) = obsanl(no) + obs_wgt(no)%wgt(l,m,n)*xa(i+l,j+m,k+n)
    end do
    end do
    end do
    obserr(no) = obsval(no) - obsanl(no)
    cost = cost + obserr(no)*obserr(no)

   !print *, 'no,obslon(no),obslat(no),obsprs(no),obsval(no),obsanl(no)=', &
   !          no,obslon(no),obslat(no),obsprs(no),obsval(no),obsanl(no)
  end do

  return
end subroutine forward

!-----------------------------------------------------------------------------------------
subroutine backward(step, iflag)
  use nn4da3d
  implicit none
  real,    intent(in)  :: step
  integer, intent(out) :: iflag

  integer :: i, j, k, l, m, n, is, js, ks, no
  real :: errwgt

  iflag = 0

  do k=0, nprs+1
  do j=0, nlat+1
  do i=0, nlon+1
    wgt(i,j,k)%bias_grd = 0.0
    wgt(i,j,k)%bias_cnt = 0

    do n=1, 3
    do m=1, 3
    do l=1, 3
      wgt(i,j,k)%grd(l,m,n) = 0.0
      wgt(i,j,k)%cnt(l,m,n) = 0
    end do
    end do
    end do
  end do
  end do
  end do

  do no=1, nobs
    is = obs_wgt(no)%ibase
    js = obs_wgt(no)%jbase
    ks = obs_wgt(no)%kbase
   !print *, 'no, is,js,ks, sum(obs_wgt(no)%wgt) = ', no, is,js,ks, sum(obs_wgt(no)%wgt)
    do n=1, 2
    do m=1, 2
    do l=1, 2
      errwgt = obserr(no)*obs_wgt(no)%wgt(l,m,n)
      wgt(is+l,js+m,ks+n)%bias_grd = wgt(is+l,js+m,ks+n)%bias_grd + errwgt
      wgt(is+l,js+m,ks+n)%bias_cnt = wgt(is+l,js+m,ks+n)%bias_cnt + 1

      do k=1, 3
      do j=1, 3
      do i=1, 3
        wgt(is+l,js+m,ks+n)%grd(i,j,k) = wgt(is+l,js+m,ks+n)%grd(i,j,k) &
                                       + errwgt*xb(is+l+i-2,js+m+j-2,ks+n+k-2)
        wgt(is+l,js+m,ks+n)%cnt(i,j,k) = wgt(is+l,js+m,ks+n)%cnt(i,j,k) + 1
      end do
      end do
      end do

     !print *, 'no, l, m, n, is+l,js+m,ks+n = ', no, l, m, n, is+l,js+m,ks+n
     !print *, 'wgt(is+l,js+m,ks+n)%grd = ', wgt(is+l,js+m,ks+n)%grd
     !print *, 'wgt(is+l,js+m,ks+n)%cnt = ', wgt(is+l,js+m,ks+n)%cnt
    end do
    end do
    end do
  end do

  do k=1, nprs
  do j=1, nlat
    i=0
    if(wgt(i,j,k)%bias_cnt > 1) then
      wgt(nlon,j,k)%bias_grd = wgt(nlon,j,k)%bias_grd + wgt(i,j,k)%bias_grd
      wgt(nlon,j,k)%bias_cnt = wgt(nlon,j,k)%bias_cnt + wgt(i,j,k)%bias_cnt

      do n=1, 3
      do m=1, 3
      do l=1, 3
        wgt(nlon,j,k)%grd(l,m,n) = wgt(nlon,j,k)%grd(l,m,n) + wgt(i,j,k)%grd(l,m,n)
        wgt(nlon,j,k)%cnt(l,m,n) = wgt(nlon,j,k)%cnt(l,m,n) + wgt(i,j,k)%cnt(l,m,n)
      end do
      end do
      end do
    end if

    i=nlon+1
    if(wgt(i,j,k)%bias_cnt > 1) then
      wgt(1,j,k)%bias_grd = wgt(1,j,k)%bias_grd + wgt(i,j,k)%bias_grd
      wgt(1,j,k)%bias_cnt = wgt(1,j,k)%bias_cnt + wgt(i,j,k)%bias_cnt

      do n=1, 3
      do m=1, 3
      do l=1, 3
        wgt(1,j,k)%grd(l,m,n) = wgt(1,j,k)%grd(l,m,n) + wgt(i,j,k)%grd(l,m,n)
        wgt(1,j,k)%cnt(l,m,n) = wgt(1,j,k)%cnt(l,m,n) + wgt(i,j,k)%cnt(l,m,n)
      end do
      end do
      end do
    end if
  end do
  end do

  do k=1, nprs
  do j=1, nlat
  do i=1, nlon
    if(wgt(i,j,k)%bias_cnt > 1) then
      wgt(i,j,k)%bias_grd = wgt(i,j,k)%bias_grd / wgt(i,j,k)%bias_cnt
    end if

    wgt(i,j,k)%bias = wgt(i,j,k)%bias + step*wgt(i,j,k)%bias_grd

    do n=1, 3
    do m=1, 3
    do l=1, 3
      if(wgt(i,j,k)%cnt(l,m,n) > 1) then
        wgt(i,j,k)%grd(l,m,n) = wgt(i,j,k)%grd(l,m,n) / wgt(i,j,k)%cnt(l,m,n)
      end if
      wgt(i,j,k)%wgt(l,m,n) = wgt(i,j,k)%wgt(l,m,n) + step*wgt(i,j,k)%grd(l,m,n)
    end do
    end do
    end do
  end do
  end do
  end do

  return
end subroutine backward

!-----------------------------------------------------------------------------------------
subroutine light_smooth(mlon, mlat, mprs, xo)
  use nn4da3d
  implicit none

  integer, intent(in) :: mlon, mlat, mprs
  real, dimension(mprs, mlat, mlon), intent(out) :: xo

  real, dimension(-1:nlon+2, -1:nlat+2, -1:nprs+2) :: xd
  integer :: i, j, k, n

  do k=1, nprs
  do j=1, nlat
  do i=1, nlon
    xd(i,j,k) = xa(i,j,k) - xb(i,j,k)
  end do
  end do
  end do

  do n = 1, 4
    call extend_boundary(nlon, nlat, nprs, xd)
    do k=1, nprs
    do j=2, nlat-1
    do i=1, nlon
      if(wgt(i,j,k)%bias_cnt < 1) then
         xd(i,j,k) = 0.25*(xd(i-1,j,k) + xd(i+1,j,k)) + 0.5*xd(i,j,k)
         xd(i,j,k) = 0.25*(xd(i,j-1,k) + xd(i,j+1,k)) + 0.5*xd(i,j,k)
      end if
    end do
    do i=nlon, 1, -1
      if(wgt(i,j,k)%bias_cnt < 1) then
         xd(i,j,k) = 0.25*(xd(i-1,j,k) + xd(i+1,j,k)) + 0.5*xd(i,j,k)
         xd(i,j,k) = 0.25*(xd(i,j-1,k) + xd(i,j+1,k)) + 0.5*xd(i,j,k)
      end if
    end do
    end do

    do j=nlat-1, 2, -1
    do i=1, nlon
      if(wgt(i,j,k)%bias_cnt < 1) then
         xd(i,j,k) = 0.25*(xd(i-1,j,k) + xd(i+1,j,k)) + 0.5*xd(i,j,k)
         xd(i,j,k) = 0.25*(xd(i,j-1,k) + xd(i,j+1,k)) + 0.5*xd(i,j,k)
      end if
    end do
    do i=nlon, 1, -1
      if(wgt(i,j,k)%bias_cnt < 1) then
         xd(i,j,k) = 0.25*(xd(i-1,j,k) + xd(i+1,j,k)) + 0.5*xd(i,j,k)
         xd(i,j,k) = 0.25*(xd(i,j-1,k) + xd(i,j+1,k)) + 0.5*xd(i,j,k)
      end if
    end do
    end do
    end do
  end do

  do k=1, mprs
  do j=1, mlat
  do i=1, mlon
    xo(k,j,i) = (xd(i,j,k) + xb(i,j,k))*scalefactor + offset
  end do
  end do
  end do

  return
end subroutine light_smooth

!-----------------------------------------------------------------------------------------
subroutine heavy_smooth(mlon, mlat, mprs, xo)
  use nn4da3d
  implicit none

  integer, intent(in) :: mlon, mlat, mprs
  real, dimension(mprs, mlat, mlon), intent(out) :: xo

  real, dimension(-1:nlon+2, -1:nlat+2, -1:nprs+2) :: xd
  integer :: i, j, k, n

  do k=1, nprs
  do j=1, nlat
  do i=1, nlon
    xd(i,j,k) = xa(i,j,k) - xb(i,j,k)
  end do
  end do
  end do

  do n = 1, 2
    call extend_boundary(nlon, nlat, nprs, xd)
    do k=1, nprs
    do j=2, nlat-1
    do i=1, nlon
      xd(i,j,k) = 0.25*(xd(i-1,j,k) + xd(i+1,j,k)) + 0.5*xd(i,j,k)
      xd(i,j,k) = 0.25*(xd(i,j-1,k) + xd(i,j+1,k)) + 0.5*xd(i,j,k)
    end do
    do i=nlon, 1, -1
      xd(i,j,k) = 0.25*(xd(i-1,j,k) + xd(i+1,j,k)) + 0.5*xd(i,j,k)
      xd(i,j,k) = 0.25*(xd(i,j-1,k) + xd(i,j+1,k)) + 0.5*xd(i,j,k)
    end do
    end do

    do j=nlat-1, 2, -1
    do i=1, nlon
      xd(i,j,k) = 0.25*(xd(i-1,j,k) + xd(i+1,j,k)) + 0.5*xd(i,j,k)
      xd(i,j,k) = 0.25*(xd(i,j-1,k) + xd(i,j+1,k)) + 0.5*xd(i,j,k)
    end do
    do i=nlon, 1, -1
      xd(i,j,k) = 0.25*(xd(i-1,j,k) + xd(i+1,j,k)) + 0.5*xd(i,j,k)
      xd(i,j,k) = 0.25*(xd(i,j-1,k) + xd(i,j+1,k)) + 0.5*xd(i,j,k)
    end do
    end do
    end do
  end do

  do k=1, mprs
  do j=1, mlat
  do i=1, mlon
    xo(k,j,i) = (xd(i,j,k) + xb(i,j,k))*scalefactor + offset
  end do
  end do
  end do

  return
end subroutine heavy_smooth

!-----------------------------------------------------------------------------------------
subroutine get_analysis(mlon, mlat, mprs, mobs, xo, oa)
  use nn4da3d
  implicit none

  integer, intent(in) :: mlon, mlat, mprs, mobs
  real, dimension(mprs, mlat, mlon), intent(out) :: xo
  real, dimension(mobs),             intent(out) :: oa

  integer :: i, j, k, n

  do k=1, mprs
  do j=1, mlat
  do i=1, mlon
    xo(k,j,i) = xa(i,j,k)*scalefactor + offset
  end do
  end do
  end do

  do n=1, mobs
    oa(n) = obsanl(n)*scalefactor + offset
  end do

  return
end subroutine get_analysis

!-----------------------------------------------------------------------------------------
subroutine reset_wgt(iflag)
  use nn4da3d
  implicit none
  integer, intent(out) :: iflag

  integer :: i, j, k

  do k=-1,nprs+2
  do j=-1,nlat+2
  do i=-1,nlon+2
    call random_number(wgt(i,j,k)%wgt)
    wgt(i,j,k)%wgt = 0.001*(wgt(i,j,k)%wgt - 0.5)
    wgt(i,j,k)%wgt(2,2,2) = 1.0 + wgt(i,j,k)%wgt(2,2,2)
    wgt(i,j,k)%cnt = 0
    call random_number(wgt(i,j,k)%bias)
    wgt(i,j,k)%bias = 0.001*(wgt(i,j,k)%bias - 0.5)
    wgt(i,j,k)%bias_cnt = 0
    wgt(i,j,k)%ibase = i - 2
    wgt(i,j,k)%jbase = j - 2
    wgt(i,j,k)%kbase = k - 2
  end do
  end do
  end do

  iflag = 0

  return
end subroutine reset_wgt

!-----------------------------------------------------------------------------------------
subroutine preconditioning(offseti, scalefactori, iflag)
  use nn4da3d
  implicit none

  real, intent(in) :: offseti, scalefactori
  integer, intent(out) :: iflag

  integer :: i, j, k, n

  offset = offseti
  scalefactor = scalefactori

  do k=1, nprs
  do j=1, nlat
  do i=1, nlon
    xb(i,j,k) = (xb(i,j,k) - offset)/scalefactor
  end do
  end do
  end do

  do n=1, nobs
    obsval(n) = (obsval(n) - offset)/scalefactor
  end do

  call extend_boundary(nlon, nlat, nprs, xb)

  iflag = 0

  return
end subroutine preconditioning

