using CairoMakie
# synthetic geometry
slope(x, y, αx, αy)     = @. αx * x + αy * y
bump(x, y, w, amp)      = @. amp * exp(-(x / w)^2 - (y / w)^2)
sphere(x, y, oz, r)     = @. max(sqrt(max(r^2 - x^2 - y^2, 0)) - oz, 0)
roughness(x, y, ω, amp) = @. amp * sin(ω * x) * cos(ω * y)
# finite differences
@views d_dx(A) = 0.5 .* (A[3:end, 2:end-1] .- A[1:end-2, 2:end-1])
@views d_dy(A) = 0.5 .* (A[2:end-1, 3:end] .- A[2:end-1, 1:end-2])
@views inn(H)  = H[2:end-1, 2:end-1]
# SIA fluxes
function flux(H, S, As, A, npow, dx, dy)
    # surface gradient
    ∇S = # ...
    # diffusivity
    D = # ...
    # flux
    qx = # ...
    qy = # ...
    return # ...
end
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
B = slope(x, y', 0.1, 0.1) +
    roughness(x, y', 5π / lx, 0.025lx) +
    bump(x, y', 0.4lx, 0.1lx)
# ice thickness
H = sphere(x, y', 0.6lx, 0.7lx)
# ice surface
S = B .+ H
# slipperiness
As = fill(As₀, nx - 2, ny - 2)
# flux
q = flux(H, S, As, A, npow, dx, dy)
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
