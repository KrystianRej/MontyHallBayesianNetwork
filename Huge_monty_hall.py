import math
from pomegranate import *

# create the discrete distribution for our friend first.
friend = DiscreteDistribution( { True: 0.5, False: 0.5 } )


# The emissions for our guest are completely random.
guest = ConditionalProbabilityTable(
	[[ True, 'A', 0.50 ],
	 [ True, 'B', 0.25 ],
	 [ True, 'C', 0.25 ],
	 [ False, 'A', 0.0 ],
	 [ False, 'B', 0.7 ],
	 [ False, 'C', 0.3 ]], [friend] )


# Then the distribution for the remaining cars.
remaining = DiscreteDistribution( { 0: 0.1, 1: 0.7, 2: 0.2, } )


# The probability of whether the prize is randomized is dependent on the number of remaining cars.
randomize = ConditionalProbabilityTable(
	[[ 0, True , 0.05 ],
     [ 0, False, 0.95 ],
     [ 1, True , 0.8 ],
     [ 1, False, 0.2 ],
     [ 2, True , 0.5 ],
     [ 2, False, 0.5 ]], [remaining] )

# Now the conditional probability table for the prize. This is dependent on the guest's friend and whether or not it is randomized.
prize = ConditionalProbabilityTable(
	[[ True, True, 'A', 0.3 ],
	 [ True, True, 'B', 0.4 ],
	 [ True, True, 'C', 0.3 ],
	 [ True, False, 'A', 0.2 ],
	 [ True, False, 'B', 0.4 ],
	 [ True, False, 'C', 0.4 ],
	 [ False, True, 'A', 0.1 ],
	 [ False, True, 'B', 0.9 ],
	 [ False, True, 'C', 0.0 ],
	 [ False, False, 'A', 0.0 ],
	 [ False, False, 'B', 0.4 ],
	 [ False, False, 'C', 0.6]], [randomize, friend] )


# Finally we can create the conditional probability table for our Monty. This is dependent on the guest and the prize.

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


# Now we can create the states for our bayesian network.


s0 = State( friend, name="friend")
s1 = State( guest, name="guest" )
s2 = State( prize, name="prize" )
s3 = State( monty, name="monty" )
s4 = State( remaining, name="remaining" )
s5 = State( randomize, name="randomize" )


# Now we'll create our bayesian network with an instance of BayesianNetwork, then add the possible states.

network = BayesianNetwork( "Advanced Bayesian Netowrk" )
network.add_states(s0, s1, s2, s3, s4, s5)


# Then the possible transitions.
network.add_transition( s0, s1 )
network.add_transition( s1, s3 )
network.add_transition( s2, s3 )
network.add_transition( s4, s5 )
network.add_transition( s5, s2 )
network.add_transition( s0, s2 )

# With a "bake" to finalize the structure of our network.
network.bake()


# Now let's create our network from the following data.

data = [[ True,  'A', 'A', 'C', 1, True  ],
		[ True,  'A', 'A', 'C', 0, True  ],
		[ False, 'A', 'A', 'B', 1, False ],
		[ False, 'A', 'A', 'A', 2, False ],
		[ False, 'A', 'A', 'C', 1, False ],
		[ False, 'B', 'B', 'B', 2, False ],
		[ False, 'B', 'B', 'C', 0, False ],
		[ True,  'C', 'C', 'A', 2, True  ],
		[ True,  'C', 'C', 'C', 1, False ],
		[ True,  'C', 'C', 'C', 0, False ],
		[ True,  'C', 'C', 'C', 2, True  ],
		[ True,  'C', 'B', 'A', 1, False ]]

network.fit( data )

# We can see the results below. Lets look at the distribution for our Friend first.
print('Distribution for Friend')
print(friend)
print('Guest')
print(guest)
print('Remaining Cars')
print(remaining)
print('Probability the prize is randomized')
print(randomize)
print('Distribution of the Prize')
print(prize)
print('Final Monty Hall')
print(monty)