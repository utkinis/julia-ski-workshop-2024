using Enzyme
# scalar functions
f(x, ω, amp) = amp * sin(ω * x)
# exact derivative
∂f_∂x_exact(x, ω, amp) = amp * ω * cos(ω * x)
# Enzyme derivative
∂f_∂x_ad(x, ω, amp)    = Enzyme.autodiff(Enzyme.Reverse, f, x, ω, amp)
# finite differences
function ∂f_∂x_fd(x, ω, amp)
    dx = x * sqrt(eps())
    0.5 * (f(x + dx, ω, amp) - f(x - dx, ω, amp)) / dx
end
# value
f(1.0, 2π, 2.0)
# derivatives
∂f_∂x_exact(1.0, 2π, 1.0)
∂f_∂x_ad(Active(1.0), Const(2π), Const(1.0))
∂f_∂x_fd(1.0, 2π, 1.0)
# mutating functions
function f!(r, x, ω, amp)
    for ix in eachindex(x)
        @inbounds r[ix] = amp * sin(ω * x[ix])
    end
    return
end
# array of inputs
x = LinRange(0, 1, 100) |> collect
# preallocate outputs
r = zeros(size(x))
# values
f!(r, x, 2π, 1.0)
# storage for propagating derivatives
r̄ = ones(size(r))
x̄ = zeros(size(x))
# reverse-mode AD, aka pullback, aka backpropagation, aka vjp
∇f!(r, x, ω, amp) = Enzyme.autodiff(Enzyme.Reverse, f!, r, x, ω, amp)
∇f!(Duplicated(r, r̄), Duplicated(x, x̄), Const(2π), Const(1.0))
@assert x̄ ≈ ∂f_∂x_exact.(x, 2π, 1.0)
# gradients of salar-valued functions
function loss(q, q_obs)
    s = 0.0
    for ix in eachindex(q)
        s += (q[ix] - q_obs[ix]) ^ 2
    end
    return s
end
# "modeled" value
q     = sin.(x)
# "observed" value
q_obs = 10.0 .* sin.(x)
# loss
@show loss(q, q_obs)
# gradient
∇loss(q, q_obs) = Enzyme.autodiff(Enzyme.Reverse, loss, q, q_obs)
# allocate storage for gradient
q̄ = zeros(size(q))
∇loss(Duplicated(q, q̄), Const(q_obs))
@assert q̄ ≈ 2.0 .* (q .- q_obs)
