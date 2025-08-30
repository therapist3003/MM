# 1.
import random as rand
import matplotlib.pyplot as plt

n = 100

a0_values = [rand.uniform(0.5, 1) for _ in range(n)]

def dynamic_system(r_vals, a0_vals=a0_values):
    sequence = []

    for i in range(len(a0_vals)):
        seq = [a0_vals[i]]
        
        for _n in range(1, n):
            a_n = (r_vals[i]**(_n-1)) * a0_vals[i]
            seq.append(a_n)

        sequence.append(seq)
    
    return sequence

# Case i) r=0
r_values = [0 for _ in range(n)]
res = dynamic_system(r_values)

n_values = [i for i in range(0, n)]

plt.title("Case i) r=0")
for seq in res:
    plt.plot(n_values, seq)
plt.savefig("r=0")
plt.show()

# Case ii) 0<r<1
r_values = [rand.uniform(0.0001, 1) for _ in range(n)]
res = dynamic_system(r_values)

plt.title("Case ii) 0<r<1")
for seq in res:
    plt.plot(n_values, seq)
plt.savefig("0<r<1")
plt.show()

# Case iii) -1 < r < 0
r_values = [rand.uniform(-0.9999, 0) for _ in range(n)]
res = dynamic_system(r_values)

plt.title("Case iii) -1<r<0")
for seq in res:
    plt.plot(n_values, seq)
plt.savefig("-1<r<0")
plt.show()

# Case iv) |r|>1
pos_r_vals = [rand.uniform(1.001, 2) for _ in range(n)]
neg_r_vals = [rand.uniform(-2, -1.001) for _ in range(n)]

all_r_vals = pos_r_vals + neg_r_vals

r_values = [rand.choice(all_r_vals) for _ in range(n)]
res = dynamic_system(r_values)

plt.title("Case iv) |r|>1")
for seq in res:
    plt.plot(n_values, seq)
plt.savefig("|r|>1")
plt.show()

# 2.
import matplotlib.pyplot as plt

n = 20

daily_dosages = [0.1, 0.2, 0.3]

def dynamic_system(daily_dose):
    seq = [0]
    
    for _ in range(n):
        new_val = seq[-1] * 0.5
        seq.append(new_val + daily_dose)
    
    return seq
    
for dose in daily_dosages:
    res = dynamic_system(dose)
    plt.plot(res, label=str(dose)+" mg")

plt.savefig("Dosage_Concentrations")
plt.show()

#3.
import matplotlib.pyplot as plt
import random as rand

sizes = [500, 1000, 10_000, 1_00_000]

def plot_hist(data):
    counts, bins, _ = plt.hist(data, bins=100)
    plt.plot(bins[:-1], counts, 'r', marker='o')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    
for size in sizes:
    plt.figure(figsize=(30,15))
    
    uniform_sampled = [rand.uniform(0,1) for _ in range(size)]
    plt.subplot(2, 2, 1)
    plot_hist(uniform_sampled)
    plt.title(f"Uniform with n={size}")

    exponential_sampled = [rand.expovariate(1) for _ in range(size)]
    plt.subplot(2, 2, 2)
    plot_hist(exponential_sampled)
    plt.title(f"Exponential with n={size}")
    
    weibull_sampled = [rand.weibullvariate(1, 1.5) for _ in range(size)]
    plt.subplot(2, 2, 3)
    plot_hist(exponential_sampled)
    plt.title(f"Weibull (scale=1, shape=1.5) with n={size}")

    triangular_sampled = [rand.triangular(0, 2, mode=1) for _ in range(size)]
    plt.subplot(2, 2, 4)
    plot_hist(triangular_sampled)
    plt.title(f"Triangular (0 to 2) with n={size}")
    
    plt.savefig(f"n={size}")
    plt.show()

#4.
import random as rand

def monte_carlo_pi(n):
    count = 0
    
    for _ in range(n):
        x, y = rand.random(), rand.random()
        
        if x**2 + y**2 <= 1:
            count += 1
            
    return 4 * (count / n)

n_vals = [1000, 10_000, 1_00_000, 10_00_000]

for n_val in n_vals:
    res = monte_carlo_pi(n_val)
    print(f"Approximation for pi with {n_val} points = {res}")

# 5.
import matplotlib.pyplot as plt
import random as rand
import numpy as np

def get_news_type():
    rand_number = rand.uniform(0, 100)
    
    if rand_number in range(1, 36):
        return "GOOD"
    elif rand_number in range(36, 81):
        return "FAIR"
    else:
        return "POOR"

def get_demand(news_type):
    if news_type=="GOOD":
        expo_rand = rand.expovariate(1/50)
        return min(100, expo_rand)
    
    elif news_type=="FAIR":
        normal_rand = np.random.normal(50, 10)
        return min(max(0, normal_rand), 100)
    
    else:
        poisson_rand = np.random.poisson(50)
        return min(max(0, poisson_rand), 100)
    
N = [200, 500, 1000, 10_000]

daily_purchase_quantity = 70

cost_price = 0.3
selling_price = 0.45
salvage_value = 0.05

profit = selling_price - cost_price

for n in N:
    revenues_from_sales = []
    profit_loss_from_excess_demand = []
    salvages_from_scrap = []
    daily_profits = []
    
    for day in range(n):
        news_type = get_news_type()
        demand = get_demand(news_type)

        # Loss case
        if demand >= daily_purchase_quantity:
            revenue = daily_purchase_quantity*selling_price
            revenues_from_sales.append(revenue)
            
            loss = (demand - daily_purchase_quantity)*profit
            profit_loss_from_excess_demand.append(loss)
            
            salvages_from_scrap.append((demand - daily_purchase_quantity)*salvage_value)
            
            daily_profits.append(revenue - (cost_price*daily_purchase_quantity) - loss)
            
        else:
            revenue = demand * selling_price
            revenues_from_sales.append(revenue)
            
            profit_loss_from_excess_demand.append(0)
            
            salvage = (daily_purchase_quantity - demand)*salvage_value
            salvages_from_scrap.append(salvage)
            
            daily_profits.append(revenue - (cost_price * demand) + salvage)
            
    plt.subplot(2, 2, 1)
    plt.title("Revenue")
    plt.plot(revenues_from_sales, color='g')
    plt.xlabel("Day")
    plt.ylabel("Amount ($)")
    
    plt.subplot(2, 2, 2)
    plt.title("Loss of profit due to excess demand")
    plt.plot(profit_loss_from_excess_demand, color='r')
    plt.xlabel("Day")
    plt.ylabel("Amount ($)")

    plt.subplot(2, 2, 3)
    plt.title("Salvage income")
    plt.plot(salvages_from_scrap, color='y')
    plt.xlabel("Day")
    plt.ylabel("Amount ($)")
    
    plt.subplot(2, 2, 4)
    plt.title("Daily profit")
    plt.plot(daily_profits, color='darkgreen')
    plt.xlabel("Day")
    plt.ylabel("Amount ($)")
    
    plt.savefig(f"n={n}")
    plt.show()

# 6.
import matplotlib.pyplot as plt
import numpy as np
import random as rand

def get_arrival_time():
    return rand.expovariate(1 / 10)

def get_service_time():
    return np.random.poisson(10)

SIM_TIME = 1000

inter_arrival_times = []
arrival_times = [] # Timestamp
service_times = []
service_begin_times = [] # Timestamp
wait_times = []
service_end_times = [] # Timestamp
time_spent_in_system = []
server_idle_times = [] 

customers_waiting = []

while True:
    if not len(inter_arrival_times):
        inter_arrival_time = 0
        arrival_time = 0
    else:
        inter_arrival_time = get_arrival_time()
        arrival_time = arrival_times[-1] + inter_arrival_time
    
    if (arrival_time >= 1000):
        break
    
    inter_arrival_times.append(inter_arrival_time)
    arrival_times.append(arrival_time)
    
    service_time = get_service_time()
    service_times.append(service_time)
    
    if len(service_end_times):
        service_begin_time = max(service_end_times[-1], arrival_time)
    else:
        service_begin_time = 0
    service_begin_times.append(service_begin_time)
        
    wait_time = service_begin_time - arrival_time
    wait_times.append(wait_time)
    
    service_end_time = service_begin_time + service_time
    service_end_times.append(service_end_time)
    
    time_spent = wait_time + service_time
    time_spent_in_system.append(time_spent)
    
    if len(service_end_times) == 1:
        idle_time = 0
    else:
        idle_time = max(0, arrival_time - service_end_times[-2])
    server_idle_times.append(idle_time)
    
avg_time_customer_waits = sum(wait_times) / len(wait_times)
print("Avg time customer waits = ", avg_time_customer_waits)

# Calculating queue lengths at end of each minute
for i in range(1000):
    
    arrived_count = sum(1 for arrival_time in arrival_times if arrival_time < i)
    serviced_count = sum(1 for service_end_time in service_end_times if service_end_time < i)

    customers_waiting.append(arrived_count - serviced_count)

avg_queue_length = sum(customers_waiting) / SIM_TIME
print("Avg queue length = ", avg_queue_length)

avg_utilization = (SIM_TIME - sum(server_idle_times)) / SIM_TIME
print("Avg utilization = ", avg_utilization)

plt.plot(customers_waiting, color='orange')
plt.title("Customers waiting in queue")
plt.xlabel("Minute")
plt.ylabel("No. of customers waiting")
plt.show()