from collections import defaultdict
from marketsim.simulator.melo_simulator import MELOSimulatorSampledArrival
from tqdm import tqdm
from agent.agent import Agent
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
surpluses = []

cumulative_payoffs = defaultdict(float)
fundamentals_aggregated = []
cum_best_buys = []
cum_best_asks = []
cum_melo_trade = []
cum_buy_cancelled = []
cum_sell_cancelled = []
cum_removed_eligibility = []
cum_removed_activation = []
cum_removed_active = []
cum_fundamental_estimates = []
cum_num_orders = []
positions = defaultdict(float)
remaining_orders = []
REPS = 2

# MOBI = 10
MOBI = 0
ZI = 1

for _ in tqdm(range(REPS)):
    fundamentals = []
    sim = MELOSimulatorSampledArrival(num_background_agents=MOBI + ZI,
                                      sim_time=1000,
                                      lam=6e-3, 
                                      mean=1e6, 
                                      num_strategic=MOBI,
                                      lam_melo=1e-3,
                                      r=0.0001, 
                                      shock_var=1e6, 
                                      q_max=10,
                                      num_zi=ZI,
                                      num_hbl=0,
                                      pv_var=5e6,
                                      shade=[10,30])
    sim.run()
    payoffs, positions_run, best_buy, best_ask, midpoint_prices, melo_trade, hbl_trades, buy_cancelled, sell_cancelled, elig, activation, active, melo_orders, fundamental_estimates, num_orders_placed, elig, activation, active= sim.end_sim()  # Dictionary {agent_id: payoff}
    # sim_dict = sim.end_sim()
    # print(sim_dict)
    # payoffs
    # positions_run
    # best_buy
    # best_ask
    # melo_trade
    # hbl_trades
    # buy_cancelled
    # sell_cancelled
    # elig
    # activation
    # active
    # melo_orders
    # fundamental_estimates
    # num_orders_placed
    # elig
    # activation,
    # active

    fundamentals.append(sim.fundamentals)
    cum_fundamental_estimates.append(fundamental_estimates)
    cum_num_orders.append(num_orders_placed)
    midpoint_prices = (np.array(best_buy) + np.array(best_ask)) / 2
    remaining_orders.append(elig + activation + active)

    # Create figure and axis
    # plt.figure(figsize=(12, 6))

    # Plot midpoint prices
    # plt.plot(midpoint_prices, label='Midpoint Price', color='blue', alpha=0.7)
    
    # plt.plot(fundamental_estimates, label="Estimate Fundamental", color="pink", alpha=0.7)
    # plt.plot(sim.fundamentals, label="Fundamental", color="brown", alpha=0.7)

    # Separate buy and sell orders from melo_orders
    buy_orders = [order for order in melo_orders if order[2] == 1]
    sell_orders = [order for order in melo_orders if order[2] == -1]

    # Plot buy orders (green dots)
    if buy_orders:
        buy_times, buy_prices, _ = zip(*buy_orders)
        # plt.scatter(buy_times, buy_prices, color='green', label='MELO Buy', alpha=0.7, s=10)

    # Plot sell orders (orange dots)
    if sell_orders:
        sell_times, sell_prices, _ = zip(*sell_orders)
        # plt.scatter(sell_times, sell_prices, color='orange', label='MELO Sell', alpha=0.7, s=10)

    # Add labels and title
    # plt.xlabel('Time Step')
    # plt.ylabel('Price')
    # plt.title('Midpoint Price and MELO Orders Over Time')
    # plt.legend()

    # # Adjust layout
    # plt.tight_layout()
    # plt.grid(alpha=0.3)

    # Show plot
    # plt.show()


    # cum_best_buys.append(best_buy)
    # cum_best_asks.append(best_ask)
    # cum_melo_trade.append(melo_trade)
    # best_buy = np.array(best_buy)
    # timesteps = np.arange(10000)
    # best_ask = np.array(best_ask)

    # # Filter out invalid values
    # min_timestep = max(min(t for t, _, _ in melo_trade) - 100, 2000)
    # max_timestep = max(t for t, _, _ in melo_trade) + 100

    # # Filter data to be within the new bounds
    # valid_indices = (timesteps >= min_timestep) & (timesteps <= max_timestep)
    # timesteps = timesteps[valid_indices]
    # best_buy = best_buy[valid_indices]
    # best_ask = best_ask[valid_indices]

    # # Filter MELO trades to be within the new bounds
    # melo_trade = [(t, qty, order_type) for t, qty, order_type in melo_trade if min_timestep <= t <= max_timestep]

    # # Plot setup
    # plt.figure(figsize=(12, 6))
    # plt.plot(timesteps, best_buy, label="Best Buy", color="#D22B2B")  # Red-Orange
    # plt.plot(timesteps, best_ask, label="Best Ask", color="#197bff")  # Blue

    # # Create custom legend handles for vertical lines
    # green_line = mlines.Line2D([], [], color="green", linestyle="dotted", label="MOBI Sell Trades (-1)")
    # orange_line = mlines.Line2D([], [], color="#F28C28", linestyle="dotted", label="MOBI Buy Trades (+1)")
    # buy_line = mlines.Line2D([], [], color="#D22B2B", linestyle="-", label="Best Buy Price")
    # ask_line = mlines.Line2D([], [], color="#197bff", linestyle="-", label="Best Ask Price")

    # # Annotate MELO trades
    # for t, qty, order_type in melo_trade:
    #     if qty == 5:
    #         if order_type == -1:
    #             plt.axvline(x=t, color="green", linestyle="dotted", alpha=0.9, linewidth=2.5)
    #             # plt.text(t, max(best_buy), f"q = {qty}", color="black",
    #                 # verticalalignment="top", fontsize=16)
    #         else:
    #             plt.axvline(x=t, color="#F28C28", linestyle="dotted", alpha=0.9, linewidth=2.5)
    #             # plt.text(t, max(best_ask), f"q = {qty}", color="black", 
    #                 # verticalalignment="top", fontsize=16)

    # for t, direction, price in hbl_trades:
    #     if min_timestep <= t <= max_timestep:  # Only plot within our time window
    #         marker_color = 'purple'  # Distinct color for HBL trades
    #         marker_shape = 'o'  # Circle marker
    #         marker_size = 60  # Slightly larger than default
    #         plt.scatter(t, price, color=marker_color, marker=marker_shape, 
    #                    s=marker_size, zorder=5, label='HBL Trade' if t == hbl_trades[0][0] else "")
            
    # # Add HBL trade to legend (only once)
    # hbl_marker = mlines.Line2D([], [], color='purple', marker='o', linestyle='None',
    #                           markersize=8, label='HBL Trade')

    # # Update legend handles to include HBL trades
    # plt.legend(handles=[buy_line, ask_line, green_line, orange_line, hbl_marker], 
    #            fontsize=16)


    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)

    # # Labels and styling
    # plt.xlabel("Timestep", fontsize=16)
    # plt.ylabel("Price", fontsize=16)
    # plt.title("Effect of MOBI Trading Activity on CDA Market", fontsize=18)
    # # plt.legend(handles=[buy_line, ask_line, green_line, orange_line], fontsize=16)
    # plt.grid(True, linestyle="--", alpha=0.5)

    # plt.show()
    cum_buy_cancelled.append(len(buy_cancelled))
    cum_sell_cancelled.append(len(sell_cancelled))
    cum_removed_eligibility.append(elig)
    cum_removed_activation.append(activation)
    cum_removed_active.append(active)


    # Accumulate payoffs for each agent
    for agent_id, payoff in payoffs.items():
        cumulative_payoffs[agent_id] += payoff
        positions[agent_id] += abs(positions_run[agent_id])



# Compute the average payoff for each agent
average_payoffs = {agent_id: cumulative_payoffs[agent_id] / REPS for agent_id in cumulative_payoffs}
avg_position =  {agent_id: positions[agent_id] / REPS for agent_id in positions}

# Print or use the results as needed
print(average_payoffs)
print(avg_position)

group1_keys = [i for i in range(30) if i in average_payoffs]
group2_keys = [i for i in average_payoffs if i >= 30]

# Calculate averages
avg_group1 = sum(average_payoffs[k] for k in group1_keys) / (len(group1_keys))
try:
    avg_group2 = sum(average_payoffs[k] for k in group2_keys) / (len(group2_keys))
except:
    avg_group2 = 0

print(f"Average for keys 0-29: {avg_group1}")
print(f"Average for keys 30+: {avg_group2}")

print("TOTAL traded", sum(value for agent_id, value in avg_position.items() if agent_id >= 30) / 5)
print("AVERAGE BUY CANCELLED", np.mean(cum_buy_cancelled))
print("AVERAGE SELL CANCELLED", np.mean(cum_sell_cancelled))


print("AVG REMOVED FROM ELIG", np.mean(cum_removed_eligibility))
print("AVG REMOVED FROM ACTIVATION", np.mean(cum_removed_activation))
print("AVG REMOVED FROM ACTIVE", np.mean(cum_removed_active))
print("AVG REMAINING IN QUEUES", np.mean(remaining_orders))
print("AVG NUM ORDERS PLACED", np.mean(cum_num_orders))

averaged_fundamentals = np.mean(fundamentals, axis=0)
cum_avg_fundamentals = np.mean(cum_fundamental_estimates, axis=0)

# The result is a 1D array of length 10,000
plt.figure(figsize=(12, 6))
plt.plot(averaged_fundamentals, color='#E86100', linewidth=1.5)
plt.plot(cum_avg_fundamentals, color='blue', linewidth=1.5)

# Customize plot
plt.xlabel("Timestep", fontsize=12)
plt.ylabel("Fundamental Value", fontsize=12)
plt.title("Fundamental Value Evolution Throughout the Simulation", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
# plt.legend()
# plt.show()

# positions = []
# values = []
# melo_profits = []
# fundamental_val = sim.market.get_final_fundamental()

# for agent_id in sim.agents:
#     agent:Agent = sim.agents[agent_id]
#     value = agent.get_pos_value() + agent.position * fundamental_val + agent.cash
#     print(agent.cash, agent.position, agent.get_pos_value(), value)
#     positions.append(agent.position)
#     values.append(value)
#     melo_profits.append(agent.melo_profit)

# import matplotlib.pyplot as plt

# # print(fundamental_val)

# plt.scatter(positions, values)
# plt.xlabel('Position')
# plt.ylabel('Value')
# plt.title(f'Agent Position vs Value at fundamental {fundamental_val}')
# plt.show()
# sum(positions)



 