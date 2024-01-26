using CairoMakie, KernelAbstractions
# synthetic geometry
slope(x, y, αx, αy)     = @. αx * x + αy * y
bump(x, y, w, amp)      = @. amp * exp(-(x / w)^2 - (y / w)^2)
sphere(x, y, oz, r)     = @. max(sqrt(max(r^2 - x^2 - y^2, 0)) - oz, 0)
roughness(x, y, ω, amp) = @. amp * sin(ω * x) * cos(ω * y)
# SIA fluxes
@kernel inbounds = true function flux_k!(q, H, S, As, A, npow, dx, dy)
    ix, iy = @index(Global, NTuple)
    # finite differences
    d_dx(A) = 0.5 * (A[ix+2, iy+1] - A[ix, iy+1])
    d_dy(A) = 0.5 * (A[ix+1, iy+2] - A[ix+1, iy])
    inn(A)  = A[ix+1, iy+1]
    # surface gradient
    ∇S = sqrt((d_dx(S) / dx)^2 +
              (d_dy(S) / dy)^2)
    # diffusivity
    D = (A * inn(H)^(npow + 2) + As[ix, iy] * inn(H)^npow) * ∇S^(npow - 1)
    # flux
    qx        = D * d_dx(S) / dx
    qy        = D * d_dy(S) / dy
    q[ix, iy] = sqrt(qx^2 + qy^2)
end
# backend
backend = CPU()
# physics
lx, ly = 20.0, 20.0
npow   = 3
A      = 1.0
As₀    = 0.0
# numerics
nx, ny = 100, 100
# preprocessing
dx, dy = lx / (nx - 1), ly / (ny - 1)
x = LinRange(-lx / 2, lx / 2, nx)
y = LinRange(-lx / 2, ly / 2, ny)
# bed topography
B = KernelAbstractions.allocate(backend, Float64, nx, ny)
copyto!(B, slope(x, y', 0.1, 0.1) +
           roughness(x, y', 5π / lx, 0.025lx) +
           bump(x, y', 0.4lx, 0.1lx))
# ice thickness
H = KernelAbstractions.allocate(backend, Float64, nx, ny)
copyto!(H, sphere(x, y', 0.6lx, 0.7lx))
# ice surface
S = B .+ H
# slipperiness
As = KernelAbstractions.allocate(backend, Float64, nx, ny)
As .= As₀
# flux
flux! = flux_k!(CPU(), 256, (nx - 2, ny - 2))
q = KernelAbstractions.allocate(backend, Float64, nx - 2, ny - 2)
flux!(q, H, S, As, A, npow, dx, dy)
# plots
fig = Figure(; size=(320, 500))
ax = (surf = Axis(fig[1, 1][1, 1]; aspect=DataAspect(), title="surface", ylabel="y"),
      flux = Axis(fig[2, 1][1, 1]; aspect=DataAspect(), title="flux", xlabel="x", ylabel="y"))
limits!(ax.surf, -lx / 2, lx / 2, -ly / 2, ly / 2)
hm = (S=surface!(ax.surf, x, y, B .+ H; colormap=:roma),
      q=heatmap!(ax.flux, x, y, q; colormap=:turbo))
cnt = (outline=contour!(ax.flux, x, y, H; levels=1e-3lx:1e-3lx, color=:white),)
Colorbar(fig[1, 1][1, 2], hm.S)
Colorbar(fig[2, 1][1, 2], hm.q)
fig
