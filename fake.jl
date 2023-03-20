# Crying baby problem described in DMU book
# State: hungry = true; not hungry = false
# Action: feed = true; do nothing = false
# Observation: crying = true; not crying = false

mutable struct FakePOMDP <: POMDP{Bool, Int64, Bool}
    r_feed::Float64
    r_hungry::Float64
    p_become_hungry::Float64
    p_cry_when_hungry::Float64
    p_cry_when_not_hungry::Float64
    discount::Float64
end
BabyPOMDP(r_feed, r_hungry) = FakePOMDP(r_feed, r_hungry, 0.1, 0.8, 0.1, 0.9)
FakePOMDP() = FakePOMDP(-5., -10.)

#updater(problem::FakePOMDP) = DiscreteUpdater(problem)

POMDPs.actions(::FakePOMDP) = (true, false)
POMDPs.actionindex(::FakePOMDP, a::Bool) = a + 1
POMDPs.states(::FakePOMDP) = collect(1:10)
POMDPs.stateindex(::FakePOMDP, s::Bool) = s
POMDPs.observations(::FakePOMDP) = (true, false)
POMDPs.obsindex(::FakePOMDP, o::Bool) = o + 1

# start knowing baby is not not hungry
POMDPs.initialstate(::FakePOMDP) = BoolDistribution(0.0)
POMDPs.initialobs(m::FakePOMDP, s) = observation(m, s)

function POMDPs.transition(pomdp::FakePOMDP, s::Int64, a::Bool)
    if a # fed
        return BoolDistribution(0.0)
    elseif s # did not feed when hungry
        return BoolDistribution(1.0)
    else # did not feed when not hungry
        return BoolDistribution(pomdp.p_become_hungry)
    end
end

function POMDPs.observation(pomdp::FakePOMDP, sp::Int64)
    if sp # hungry
        return BoolDistribution(pomdp.p_cry_when_hungry)
    else
        return BoolDistribution(pomdp.p_cry_when_not_hungry)
    end
end

function POMDPs.reward(pomdp::FakePOMDP, s::Int64, a::Bool)
    r = 0.0
    if s == 10 # hungry
        r =100
    end
    return r
end

POMDPs.discount(p::FakePOMDP) = p.discount

#