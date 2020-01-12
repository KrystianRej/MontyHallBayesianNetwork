import math
from pomegranate import *

guest = DiscreteDistribution( { 'A': 1./3, 'B': 1./3, 'C': 1./3 } )
prize = DiscreteDistribution( { 'A': 1./3, 'B': 1./3, 'C': 1./3 } )

monty = ConditionalProbabilityTable(
	[[ 'A', 'A', 'A', 0.0 ],
	 [ 'A', 'A', 'B', 0.5 ],
	 [ 'A', 'A', 'C', 0.5 ],
	 [ 'A', 'B', 'A', 0.0 ],
	 [ 'A', 'B', 'B', 0.0 ],
	 [ 'A', 'B', 'C', 1.0 ],
	 [ 'A', 'C', 'A', 0.0 ],
	 [ 'A', 'C', 'B', 1.0 ],
	 [ 'A', 'C', 'C', 0.0 ],
	 [ 'B', 'A', 'A', 0.0 ],
	 [ 'B', 'A', 'B', 0.0 ],
	 [ 'B', 'A', 'C', 1.0 ],
	 [ 'B', 'B', 'A', 0.5 ],
	 [ 'B', 'B', 'B', 0.0 ],
	 [ 'B', 'B', 'C', 0.5 ],
	 [ 'B', 'C', 'A', 1.0 ],
	 [ 'B', 'C', 'B', 0.0 ],
	 [ 'B', 'C', 'C', 0.0 ],
	 [ 'C', 'A', 'A', 0.0 ],
	 [ 'C', 'A', 'B', 1.0 ],
	 [ 'C', 'A', 'C', 0.0 ],
	 [ 'C', 'B', 'A', 1.0 ],
	 [ 'C', 'B', 'B', 0.0 ],
	 [ 'C', 'B', 'C', 0.0 ],
	 [ 'C', 'C', 'A', 0.5 ],
	 [ 'C', 'C', 'B', 0.5 ],
	 [ 'C', 'C', 'C', 0.0 ]], [guest, prize] )


s1 = State( guest, name="guest" )
s2 = State( prize, name="prize" )
s3 = State( monty, name="monty" )

network = BayesianNetwork( "Bayesian Network Classic" )
network.add_states( s1, s2, s3 )

network.add_transition( s1, s3 )
network.add_transition( s2, s3 )

network.bake()

# Now we can check the possible states in our network.
print("\t".join([state.name for state in network.states]))

# Now we can see what happens to our network when our Guest chooses 'A'.
observations = { 'guest' : 'A' }
beliefs = map( str, network.predict_proba( observations ) )
print("\n".join("{}\t{}".format(state.name, belief) for state, belief in zip(network.states, beliefs)))


# Now our host chooses 'B'. (note that prize goes to 66% if you switch)
observations = { 'guest' : 'A', 'monty' : 'B' }
beliefs = map( str, network.predict_proba( observations ) )
print("\n".join("{}\t{}".format(state.name, belief) for state, belief in zip(network.states, beliefs)))

# train data
data = [[ 'A', 'A', 'C' ],
		[ 'A', 'A', 'C' ],
		[ 'A', 'A', 'B' ],
		[ 'A', 'A', 'A' ],
		[ 'A', 'A', 'C' ],
		[ 'B', 'B', 'B' ],
		[ 'B', 'B', 'C' ],
		[ 'C', 'C', 'A' ],
		[ 'C', 'C', 'C' ],
		[ 'C', 'C', 'C' ],
		[ 'C', 'C', 'C' ],
		[ 'C', 'B', 'A' ]]

network.fit( data )

# Results:
print("monty:")
print(monty)

print("prize:")
print(prize)

print("guest:")
print(guest)

