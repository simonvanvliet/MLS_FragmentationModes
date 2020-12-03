******

FIGURE 2A
- Will show dynamics over time of some replicates when there are no group events. The idea is to show how cooperators/population can be maintained with vs without group events 
- Data needed: cooperator frequency, group size, number of groups, and number of cells over time
- Number of replicates: maybe 10 per parameter combination?
- Parameters:
	gr_CFis = 0.0
	gr_SFis = 0.0
	indv_mutR = [0.001, 0.01] # trying a couple of different values to see which will look better
	Extinction rate must be 0, which I don't think can be done by just changing parameters
	{offspr_size, offspr_frac} = [{0.05, 0.01}, {0.05, 0.5}, {0.05, 0.9}, {0.25, 0.5}, {0.45, 0.5}] # trying the same points that you used for the GroupEvolution simulations.


FIGURE 2B
- Same as 2A but now there will be group events
- Number of replicates: maybe 10 per parameter combination?
- Parameters:
	gr_CFis = [0.01, 0.05, 0.1] # trying a couple of different values to see which will look better
	indv_mutR = [0.001, 0.01] # trying a couple of different values to see which will look better	
	Death rate now normal
	{offspr_size, offspr_frac} = [{0.05, 0.01}, {0.05, 0.5}, {0.05, 0.9}, {0.25, 0.5}, {0.45, 0.5}]

******

FIGURE 3A
- For the three archetypes, we show on x-axis the mutation rate, and on y-axis the population size
- Data needed: population size, mutation rate
- Number of replicates: maybe 5 per parameter combination
- Notes: for the structure of the paper to make sense, gr_SFis would ideally be zero. Based on the mutational meltdown picture, that would require us to get mutation rates in the range 0-0.5
- Parameters:
	indv_mutR = [0.00 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50]
	gr_SFis = 0
	{offspr_size, offspr_frac} = archetypes = [{0.486, 0.01}, {0.01, 0.01}, {0.248, 0.99}]


FIGURE 3B
- For each strategy, maximum mutation rate. These data already exist, they're in the mutational meltdown folder and look great. 


FIGURE 3C
- Percent decrease in fraction of cooperators for increasing values of mu for each strategy.
- I'm still not sure how to plot this because our space is 2d. Maybe you can help decide?
	- One alternative is to show the decrease in fraction of cooperators for increasing values of mu along the outer perimeter only (just like in the current figure in the Transects folder), but that could be hard to explain/justify. 
	- Another alternative would be to pick several points along the triangle and plot the fraction of cooperators (or relative fraction) as circles. Mu = 0.001 would always have the biggest circle, then over that circle would be smaller circles for the other values of mu. The circles would be most different in size away from the complete fragmentation archetype. I haven't tried out how this looks like but it would avoid doing the perimeter thing.
	- I welcome alternative suggestions.
- Number of replicates: maybe 5 per parameter combination, so we can get smooth averages
- Parameters:
	gr_SFis = 0 (to be consistent with other panels)
	indv_mutR = [0.001, 0.01, 0.1]
	{offspr_size, offspr_frac} = depending on the idea - either the perimeter - or 9 points, namely four equally spaced points along y axis, three equally spaced points at x = 0.25, and the point 0.5,0.5


FIGURE 3D, 3E
- Triangle showing, for each strategy, the number of cells and number of groups at equilibrium.
- Similar to the triangles in scanStates folder but with slope = 0
- Number of replicates: maybe 5 per parameter combination
- Parameters:
	gr_SFis = 0 (to be consistent with other panels)
	{offspr_size, offspr_frac} = triangle

******

FIGURE 4A
- Triangles showing total number of cells for different values of invd_NType and indv_asymmetry
- Not a full factorial. Keep NType = 1 when varying asymmetry, and vice-versa
- Number of replicates: maybe 5 per parameter combination.
	gr_SFis = 0
	indv_asymmetry = [1 2 3 4]
	indv_NType = [1 2 3 4]
	{offspr_size, offspr_frac} = triangle


FIGURE 4B
- x axis shows mu, y axis shows total number of cells. Three lines, for increasing number of types
- Focus on a particular strategy. I suggest the strategy that corresponds to perimeter_loc = 0.1 since it's around the optimal value.
- Number of replicates: maybe 5 per parameter combination.
	gr_SFis = 0
	indv_NType = [1 2 3]
	indv_mutR = [0.001000000 0.001668101 0.002782559 0.004641589 0.007742637 0.012915497 0.021544347 0.035938137 0.059948425 0.100000000] # those are 10 logarithmically spaced values between 0.001 and 0.1
	{offspr_size, offspr_frac} = value that corresponds to perimeter_loc = 0.1


FIGURE 4C
- Population size vs strategy - I’m thinking of focusing on the top diagonal only, since it's easier to explain. Alternatively we can go with the whole perimeter, but only if we did that also for figure 3C.
- Three mutation rates: 0.001, 0.01, 0.1
- Same that we already have in Transects folder but with more replicates so that it results in smoother lines
	gr_SFis = 0
	indv_mutR = [0.001 0.01 0.1]
	{offspr_size, offspr_frac} = perimeter or upped diagonal

******

FIGURE 5
- I don't think we need new data for this, but I will try to plot it in a more compelling way

******

FIGURE 6
- The same exact runs that we have for the current GroupEvolution directory.

******## 