using CairoMakie, Enzyme
# synthetic geometry
slope(x, y, αx, αy)     = @. αx * x + αy * y
bump(x, y, w, amp)      = @. amp * exp(-(x / w)^2 - (y / w)^2)
sphere(x, y, oz, r)     = @. max(sqrt(max(r^2 - x^2 - y^2, 0)) - oz, 0)
roughness(x, y, ω, amp) = @. amp * sin(ω * x) * cos(ω * y)
# synthetic slipperiness
function synthetic_slipperiness(B, z_sedi, z_rock, As_sedi, As_rock)
    return @. clamp(As_rock * (B - z_sedi) / (z_rock - z_sedi) +
                    As_sedi * (B - z_rock) / (z_sedi - z_rock), As_rock, As_sedi)
end
# SIA fluxes
function flux!(q, H, S, As, A, npow, dx, dy)
    for iy in axes(q, 2), ix in axes(q, 1)
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
end
# loss function
function loss(q_obs, q, H, S, As, A, npow, dx, dy, wt)
    flux!(q, H, S, As, A, npow, dx, dy)
    return wt * sum((q .- q_obs) .^ 2)
end
# gradient of loss function
function grad_loss(q_obs, q, H, S, As, A, npow, dx, dy, wt)
    Enzyme.autodiff(Enzyme.Reverse, loss, q_obs, q, H, S, As, A, npow, dx, dy, wt)
end
# physics
lx, ly = 20.0, 20.0
npow   = 3
A      = 1.0
As₀    = 45.0
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
# synthetic slipperiness
Asₛ = synthetic_slipperiness(B, 0.0lx, 0.12lx, 100.0, 0.0)
# flux
q = zeros(nx - 2, ny - 2)
flux!(q, H, S, As, A, npow, dx, dy)
# observed flux
q_obs = zeros(nx - 2, ny - 2)
flux!(q_obs, H, S, Asₛ, A, npow, dx, dy)
# loss
wt = inv(sum(q_obs .^ 2))
J(As) = loss(q_obs, q, H, S, As, A, npow, dx, dy, wt)
# gradient of loss
Ās = similar(As)
q̄  = similar(q)
function ∇J!(Ās, As)
    q̄  .= 0.0
    Ās .= 0.0
    grad_loss(Const(q_obs),
              Duplicated(q, q̄),
              Const(H),
              Const(S),
              Duplicated(As, Ās),
              Const(A),
              Const(npow),
              Const(dx),
              Const(dy),
              Const(wt))
end
∇J!(Ās, As)
# plots
fig = Figure(; size=(920, 500))
ax = (surf=Axis(fig[1, 1][1, 1]; aspect=DataAspect(), title="surface", ylabel="y"),
      flux=(obs=Axis(fig[2, 2][1, 1]; aspect=DataAspect(), title="flux (observed)", xlabel="x", ylabel="y"),
            mod=Axis(fig[2, 1][1, 1]; aspect=DataAspect(), title="flux (modeled)", xlabel="x")),
      slip=Axis(fig[1, 2][1, 1]; aspect=DataAspect(), title="synthetic slipperiness"),
      grad=Axis(fig[1, 3][1, 1]; aspect=DataAspect(), title="∇J"),
      diff=Axis(fig[2, 3][1, 1]; aspect=DataAspect(), title="q-qₒ"))
limits!(ax.surf, -lx / 2, lx / 2, -ly / 2, ly / 2)
hm = (S=surface!(ax.surf, x, y, B .+ H; colormap=:roma),
      q=(obs=heatmap!(ax.flux.obs, x, y, q_obs; colormap=:turbo),
         mod=heatmap!(ax.flux.mod, x, y, q; colormap=:turbo)),
      Asₛ=heatmap!(ax.slip, x, y, Asₛ; colormap=:vik),
      Ās=heatmap!(ax.grad, x, y, Ās; colormap=:turbo),
      dq=heatmap!(ax.diff, x, y, q .- q_obs; colormap=:turbo))
cnt = (outline=(contour!(ax.surf, x, y, H; levels=1e-3lx:1e-3lx, color=:white),
                contour!(ax.slip, x, y, H; levels=1e-3lx:1e-3lx, color=:white),
                contour!(ax.flux.obs, x, y, H; levels=1e-3lx:1e-3lx, color=:white),
                contour!(ax.flux.mod, x, y, H; levels=1e-3lx:1e-3lx, color=:white),
                contour!(ax.grad, x, y, H; levels=1e-3lx:1e-3lx, color=:white),
                contour!(ax.diff, x, y, H; levels=1e-3lx:1e-3lx, color=:white)),)
Colorbar(fig[1, 1][1, 2], hm.S)
Colorbar(fig[2, 2][1, 2], hm.q.obs)
Colorbar(fig[2, 1][1, 2], hm.q.mod)
Colorbar(fig[1, 2][1, 2], hm.Asₛ)
Colorbar(fig[1, 3][1, 2], hm.Ās)
Colorbar(fig[2, 3][1, 2], hm.dq)
display(fig)