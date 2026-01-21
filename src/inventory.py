
import numpy as np
import pandas as pd
from collections import deque

class InventorySimulation:
    def __init__(self, initial_stock, expiry_days, lead_time=1, review_period=1):
        """
        initial_stock: Total starting inventory (assumed fresh for simplicity or generic)
        expiry_days: Shelf life in days
        """
        # Inventory State: list of batches (expiry_date, quantity)
        # Assuming initial stock has full shelf life relative to start
        self.inventory = deque() 
        # For simplicity, we track 'days_remaining' instead of absolute dates to avoid complex calendaring in inner loop
        # Or relative mapping.
        self.inventory.append({'days_left': expiry_days, 'qty': initial_stock})
        
        self.expiry_days = expiry_days
        self.lead_time = lead_time
        self.orders = [] # (arrival_day, qty)
        
        # Metrics
        self.lost_sales = 0
        self.waste = 0
        self.holding_cost_accum = 0
        self.ordering_cost_accum = 0
        self.stockout_days = 0
        self.total_sold = 0
        self.total_demand = 0
        
    def step(self, demand, ordering_cost, holding_cost_per_unit, shortage_cost_per_unit, waste_cost_per_unit, eoq, rop):
        """
        Run one day of simulation.
        """
        self.total_demand += demand
        
        # 1. Receive Orders
        arrived_qty = 0
        # Check orders arriving today (assuming we call step with an incremental day index or handle internally)
        # Simplified: We'll manage 'days_until_arrival' in the orders list
        
        new_orders = []
        for days_wait, qty in self.orders:
            if days_wait == 0:
                arrived_qty += qty
                # Add to inventory with full shelf life
                self.inventory.append({'days_left': self.expiry_days, 'qty': qty})
            else:
                new_orders.append((days_wait - 1, qty))
        self.orders = new_orders
        
        # 2. Daily Aging & Expiry Removal
        # Decrease days_left for all batches
        # Remove if days_left < 0 (expired)
        expired_today = 0
        valid_inventory = deque()
        
        current_stock = 0
        
        # We need to age *before* or *after* consumption? 
        # Usually: Start of Day -> Receive -> Fullfill -> End of Day (Age)
        # Let's do: Receive -> Fulfill -> Age
        
        # Calculate current available stock for fulfillment
        for batch in self.inventory:
            current_stock += batch['qty']
            
        # 3. Fulfill Demand (FEFO)
        # Sort by days_left (ascending) - deque is naturally appending new stuff at end, 
        # but if we had mixed batches, we should sort. 
        # Assuming FIFO/FEFO aligns with creation order if self life is constant.
        # Yes, standard FIFO is FEFO if shelf life is constant.
        
        fulfilled = 0
        shortage = 0
        
        if current_stock >= demand:
            fulfilled = demand
            rem_demand = demand
            
            while rem_demand > 0 and self.inventory:
                batch = self.inventory[0] # Oldest
                if batch['qty'] > rem_demand:
                    batch['qty'] -= rem_demand
                    rem_demand = 0
                    self.inventory[0] = batch # Update
                else:
                    rem_demand -= batch['qty']
                    self.inventory.popleft() # Fully used
        else:
            fulfilled = current_stock
            shortage = demand - current_stock
            self.inventory.clear()
            self.stockout_days += 1
            
        self.total_sold += fulfilled
        self.lost_sales += shortage
        
        # 4. Age and Expire
        # We need to iterate and age.
        # Since we modify the list, let's rebuild it or iterate carefully.
        # Rebuilding is safe.
        output_inventory = deque()
        day_waste = 0
        
        current_stock_end_of_day = 0
        
        while self.inventory:
            batch = self.inventory.popleft()
            batch['days_left'] -= 1
            if batch['days_left'] >= 0:
                output_inventory.append(batch)
                current_stock_end_of_day += batch['qty']
            else:
                day_waste += batch['qty']
        
        self.inventory = output_inventory
        self.waste += day_waste
        
        # 5. Review & Order (Continuous Review)
        # Check current inventory position + on_order
        on_order = sum([o[1] for o in self.orders])
        inventory_position = current_stock_end_of_day + on_order
        
        if inventory_position <= rop:
            # Place Order
            # Order EOQ
            self.orders.append((self.lead_time, eoq))
            self.ordering_cost_accum += ordering_cost
            
        # Costs
        self.holding_cost_accum += current_stock_end_of_day * holding_cost_per_unit
        
        return {
            'fulfilled': fulfilled,
            'shortage': shortage,
            'waste': day_waste,
            'stock': current_stock_end_of_day
        }

def run_simulation(demand_series, initial_stock, expiry_days, params):
    """
    Run simulation over a demand series.
    params: {ordering_cost, holding_cost, shortage_cost, waste_cost, lead_time, eoq, rop}
    """
    sim = InventorySimulation(initial_stock, expiry_days, lead_time=params['lead_time'])
    
    results = []
    
    for demand in demand_series:
        day_res = sim.step(
            demand=demand,
            ordering_cost=params['ordering_cost'],
            holding_cost_per_unit=params['holding_cost'],
            shortage_cost_per_unit=params['shortage_cost'],
            waste_cost_per_unit=params['waste_cost'],
            eoq=params['eoq'],
            rop=params['rop']
        )
        results.append(day_res)
        
    # Summary Metrics
    total_cost = (
        sim.ordering_cost_accum + 
        sim.holding_cost_accum + 
        (sim.lost_sales * params['shortage_cost']) + 
        (sim.waste * params['waste_cost'])
    )
    
    service_level = (sim.total_demand - sim.lost_sales) / sim.total_demand if sim.total_demand > 0 else 1.0
    
    return {
        'total_cost': total_cost,
        'service_level': service_level,
        'stockout_days': sim.stockout_days,
        'waste_units': sim.waste,
        'lost_sales_units': sim.lost_sales,
        'details': pd.DataFrame(results)
    }

def calculate_eoq_rop(mean_demand, std_demand, ordering_cost, holding_cost, lead_time, service_level_z=1.96):
    """
    Calculate generic EOQ and ROP.
    """
    if mean_demand <= 0: return 0, 0
    
    # EOQ
    # EOQ = sqrt(2 * D * S / H) -> D is annual demand usually. Here daily mean * 365? 
    # Or just period demand.
    # Text implies daily simulation. Let's stick to consistent units.
    # If holding cost is 'per day', then D is daily.
    # Usually Holding is % of value per year. 
    # Let's assume the sliders give 'Holding Cost per Unit per Day' roughly or similar.
    
    # Assume: holding_cost input is for the same period as demand (day).
    if holding_cost == 0: holding_cost = 0.001 # avoid div by zero
    
    eoq = np.sqrt((2 * mean_demand * ordering_cost) / holding_cost)
    
    # ROP = (Daily Demand * Lead Time) + Safety Stock
    # Safety Stock = Z * std_dev * sqrt(Lead Time)
    safety_stock = service_level_z * std_demand * np.sqrt(lead_time)
    rop = (mean_demand * lead_time) + safety_stock
    
    return eoq, rop
