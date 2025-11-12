
import os
from xml.parsers.expat import model
from duckdb import df
import numpy as np
import math
import pyomo.environ as pyo
import pyomo.opt as po
import pandas as pd


class optimizer:
    def __init__(self, solverpath_exe, data: pd.DataFrame,n_cycles: int, energy_cap: int, power_cap: int,no_intervals: int, 
                 soc_min:float,Pmax_aFRR:float, n_cap:int, w1: float , w2: float, update_proba:bool ):
        """Parameters:
        - n_cycles: Maximum number of allowed cycles
        - energy_cap: Energy capacity
        - power_cap: Power capacity
        - data containing paFRR_up, paFRR_down,pBM_up, pBM_down, pDAM, pIDM as 96-dimensional  price vectors, surplus and deficit
        - Pmax_aFRR: maximum coeficient for the aFRR capacity
        - no_intervals: Number of intervals in a day (96 for 15-min intervals)
        - soc_min: Minimum state of charge
        - n_cap: Maximum number of cycles for capacity used in aFRR
        - BM_Up_proba: 96-dimensional BM probabilities for activation for Up
        - BM_Down_proba: 96-dimensional BM probabilities for activation for Down
        - w1, w2: Weights for the objective function
        """
        self.solverpath_exe = solverpath_exe
        self.data=data
        # Get prices from the data
        self.pADown =  data['pADown'].values
        self.pADown[self.pADown == 0] = 0.0001 # Avoid 0 values for prices
        self.pAUp =  data['pAUp'].values
        self.pAUp[self.pAUp == 0] = 0.0001
        self.pBMDown =  data['pBMDown'].values
        self.pBMDown[self.pBMDown == 0] = 0.0001
        self.pBMUp =  data['pBMUp'].values
        self.pBMUp[self.pBMUp == 0] = 0.0001
        self.pDAM =  data['pDAM'].values
        self.pIDM =  data['pIDM'].values
        # Initialize other variables
        self.n_cycles=n_cycles
        self.energy_cap=energy_cap
        self.power_cap=power_cap
        self.no_intervals=no_intervals
        self.qBESSmax=self.power_cap / self.no_intervals
        self.soc_min=soc_min
        self.Pmax_aFRR=Pmax_aFRR
        self.n_cap=n_cap
        # Daily discharged energy limit
        self.volume_limit = (self.energy_cap-self.soc_min) * self.n_cycles

        # Get BM probabilities from the data initial values for aFRR only
        self.BM_Down_proba=self.data['proba_Down'].values
        self.BM_Up_proba=self.data['proba_Up'].values
        self.update_proba=update_proba


        # Surplus and demand
        self.S=data['S'].values
        self.D=data['D'].values
        self.D_updated=data['D_updated'].values
        self.S_updated=data['S_updated'].values
        self.TotalGeneration=data['generation'].sum()
        self.delta_D = data['delta_D'].values
        self.delta_S = data['delta_S'].values

        # Calculate the maximum revenue from DAM and IDM
        self.payment_DAM=((data['S']-data['D'])*data['pDAM']).sum() #the revenue of selling all surplus on DAM
        self.payment_IDM=((data['delta_S']-data['delta_D'])*data['pIDM']).sum()
        self.payment = self.payment_DAM + self.payment_IDM
        self.w1=w1
        self.w2=w2


        # print(f'Surplus {self.S} and demand {self.D}')
        print(f'Payment DAM {self.payment_DAM} EUR,  Payment IDM {self.payment_IDM} EURR')
        # print(f'Total Initial Payment {self.payment} EUR')

    def set_glpk_solver(self):
        """
        Sets the solver to be used to the GLPK solver.
        Note that you have to install the GLPK solver on your machine to run the optimization model.
        It is available here: https://www.gnu.org/software/glpk/
        """
        return pyo.SolverFactory("glpk", executable=self.solverpath_exe)
    
    '''Step 1 - maximizes the revenue from aFRR (with BM coefficients) for BESS and Flexibility Up and Down.'''
    def step1_optimize_aFRR(self):
        """
        Calculates optimal charge/discharge schedule on the aFRR for a given 96-d paFRR_up and paFRR_down price vectors and BM activation probability.
        
        Returns:
        - soc_aFRR: Resulting state of charge schedule after aFRR and BM
        - x_aFRR_ch: Resulting charge schedule / Positions on aFRR Down
        - x_aFRR_dis: Resulting discharge schedule / Positions on aFRR Up
        - profit_aFRR: Profit from aFRR 
        """
        
        # Initialize pyomo model:
        model = pyo.ConcreteModel()

        # Set parameters:

        # Number of hours
        model.H = pyo.RangeSet(0, 23)

        # Number of quarters
        model.Q = pyo.RangeSet(1, 96)

        # Number of quarters plus 1
        model.Q_plus_1 = pyo.RangeSet(1, 97)

        # Daily discharged energy limit
        volume_limit = (self.energy_cap-self.soc_min) * self.n_cycles
        
        # Daily discharged capacity limit
        capacity_limit= (self.energy_cap-self.soc_min) * self.n_cap
        
        # Initialize variables:
        # State of charge. SOC is expressed in MWH and ranges from 0 to energy_cap
        model.soc = pyo.Var(model.Q_plus_1, domain=pyo.Reals)

        # Capacity for charges on the aFRR. It represents the coeficient of the rated power used during the time interval
        model.cha_aFRR = pyo.Var(model.Q, domain=pyo.NonNegativeReals, bounds=(0,self.Pmax_aFRR))
        # Capacity for discharges on the aFRR. It represents the coeficient of the rated power used during the time interval
        model.dis_aFRR = pyo.Var(model.Q, domain=pyo.NonNegativeReals, bounds=(0, self.Pmax_aFRR))


        # Remark: In some of the constraints, you will notice that the indices [q] and [q-1] are used for the same quarter. This is due to Python lists counting from 0 and Pyomo Variable lists counting from 1.
    
        def set_maximum_soc(model, q):
            """
            State of charge (MWh) can never be higher than Energy Capacity. (Constraint 1.1)
            """
            return model.soc[q] <= self.energy_cap

        def set_minimum_soc(model, q):
            """
            State of charge (MWh) can never be less than SOC min. (Constraint 1.2)
            """
            return model.soc[q] >= self.soc_min

        def set_first_soc_to_0(model):
            """
            State of charge (MWh)  at the first quarter. (Constraint 1.3)
            """
            return model.soc[1] == self.energy_cap/2

        def set_last_soc_to_0(model):
            """
            State of charge (MWh) at quarter 97 (i.e., first quarter of next day) must >=soc_min. (Constraint 1.4)
            """
            return model.soc[97] >= self.soc_min

        def soc_step_constraint(model, q):
            """
            The state of charge of each quarter equals the state if charge of the previous quarter plus charges minus discharges. (Constraint 1.5)
            """
            #return model.soc[q + 1] == model.soc[q] + self.qBESSmax * (model.cha_aFRR[q]*self.BM_Down_proba[q-1] -  model.dis_aFRR[q]*self.BM_Up_proba[q-1])
            return model.soc[q + 1] == model.soc[q] + self.qBESSmax * (model.cha_aFRR[q] -  model.dis_aFRR[q])

        def charge_cycle_limit(model):
            """
            Sum of all charges has to be below the daily limit. (Constraint 1.6)
            """
            return sum(model.cha_aFRR[q] * self.qBESSmax for q in model.Q) <= volume_limit

        def discharge_cycle_limit(model):
            """
            Sum of all discharges has to be below the daily limit. (Constraint 1.7)
            """
            return sum(model.dis_aFRR[q] * self.qBESSmax for q in model.Q) <= volume_limit

        
        def capacity_charge_cycle_limit(model):
            """
            Sum of all charges for capacity has to be below the daily limit. (Constraint 1.8)
            """
            return sum(model.cha_aFRR[q] * self.qBESSmax for q in model.Q) <= capacity_limit

        def capacity_discharge_cycle_limit(model):
            """
            Sum of all discharges for capacity has to be below the daily limit. (Constraint 1.9)
            """
            return sum(model.dis_aFRR[q] * self.qBESSmax for q in model.Q) <= capacity_limit
        
        # Apply constraints on the model:
        model.set_maximum_soc = pyo.Constraint(model.Q_plus_1, rule=set_maximum_soc)
        model.set_minimum_soc = pyo.Constraint(model.Q_plus_1, rule=set_minimum_soc)
        model.set_first_soc_to_0 = pyo.Constraint(rule=set_first_soc_to_0)
        model.set_last_soc_to_0 = pyo.Constraint(rule=set_last_soc_to_0)
        model.soc_step_constraint = pyo.Constraint(model.Q, rule=soc_step_constraint)
        #model.charge_cycle_limit = pyo.Constraint(rule=charge_cycle_limit)
        #model.discharge_cycle_limit = pyo.Constraint(rule=discharge_cycle_limit)
        model.capacity_charge_cycle_limit = pyo.Constraint(rule=capacity_charge_cycle_limit)
        model.capacity_discharge_cycle_limit = pyo.Constraint(rule=capacity_discharge_cycle_limit)


        # Define objective function and solve the optimization problem.
        # The objective is to maximize revenue from aFRR trades over all possible charge-discharge schedules.
        #f_aFRR_BESS = sum(self.qBESSmax * (self.pAUp[q-1]*model.dis_aFRR[q] + self.pADown[q-1]*model.cha_aFRR[q] + self.pBMUp[q-1]*self.BM_Up_proba[q-1]*model.dis_aFRR[q] + self.pBMDown[q-1]*self.BM_Down_proba[q-1]*model.cha_aFRR[q]) for q in model.Q)
        f_aFRR_BESS = sum(self.qBESSmax * (self.pAUp[q-1]*model.dis_aFRR[q] + self.pADown[q-1]*model.cha_aFRR[q]) for q in model.Q)
        model.obj = pyo.Objective(expr=f_aFRR_BESS, sense=pyo.maximize)

        solver = self.set_glpk_solver()
        solver.solve(model, timelimit=5)

        # Retrieve arrays of resulting optimal soc/charge/discharge schedules after the DA Auction:
        # BESS
        self.soc_aFRR = [model.soc[q].value for q in range(1, len(self.pAUp) + 1)]
        self.x_aFRR_ch = [model.cha_aFRR[q].value for q in range(1, len(self.pADown) + 1)]
        self.x_aFRR_dis = [model.dis_aFRR[q].value for q in range(1, len(self.pAUp) + 1)]
        # Get BM BESS initial activation using 90% probabilities
        self.x_BM_ch=np.array(self.x_aFRR_ch, dtype=float)
        self.x_BM_dis=np.array(self.x_aFRR_dis, dtype=float)

        # Calculate profit from aFRR trades:
        self.profit_aFRR_BESS = sum([self.qBESSmax * (self.pAUp[q]*self.x_aFRR_dis[q] + self.pADown[q]*self.x_aFRR_ch[q]) for q in range(len(self.pADown))])

        return(self.soc_aFRR, self.x_aFRR_ch, self.x_aFRR_dis, self.x_BM_ch,self.x_BM_dis)

    def step2_optimize_DAM(self):
        """
        Calculates optimal charge/discharge schedule on the day-ahead auction (daa) for a given 96-d pDAM vector.

        Returns:
        - soc_DAM: Resulting state of charge schedule after DAM
        - x_DAM_ch: Resulting charge schedule / Positions on DAM Auction
        - x_DAM_dis: Resulting discharge schedule / Positions on DAM Auction
        - x_EC_ch1: Resulting charges from EC to reduce the surplus
        - x_EC_dis1: Resulting discharge from EC to reduce the demand
        - profit_DAM: Profit from Day-ahead auction trades
        - x_cha1: Combined charges from BM+DAM+EC as coeficients of the rated power
        - x_dis1: Combined discharges from BM+DAM+EC as coeficients of the rated power
        """

        # Initialize pyomo model:
        model = pyo.ConcreteModel()

        # Set parameters:

        # Number of hours
        model.H = pyo.RangeSet(0, 23)

        # Number of quarters
        model.Q = pyo.RangeSet(1, 96)

        # Number of quarters plus 1
        model.Q_plus_1 = pyo.RangeSet(1, 97)

        # Initialize variables:

        # State of charge. SOC is expressed in MWH and ranges from 0 to energy_cap
        model.soc = pyo.Var(model.Q_plus_1, domain=pyo.Reals)

        # Charges on the Day-ahead auction. It represents the coeficient of the rated power used during the time interval
        model.x_DAM_ch = pyo.Var(model.Q, domain=pyo.NonNegativeReals, bounds=(0, 1))

        # Discharges on the Day-ahead auction. It represents the coeficient of the rated power used during the time interval
        model.x_DAM_dis = pyo.Var(model.Q, domain=pyo.NonNegativeReals, bounds=(0, 1))
        
        # Charges on EC to reduce the surplus
        model.x_EC_ch1 = pyo.Var(model.Q, domain=pyo.NonNegativeReals, bounds=(0, 1))

        # Discharges on EC to reduce the demand
        model.x_EC_dis1 = pyo.Var(model.Q, domain=pyo.NonNegativeReals, bounds=(0, 1))

        # Define Constraints:
        # Remark: In some of the constraints, you will notice that the indices [q] and [q-1] are used for the same quarter. This is due to Python lists counting from 0 and Pyomo Variable lists counting from 1.
    
        def set_maximum_soc(model, q):
            """
            State of charge (MWh) can never be higher than Energy Capacity. (Constraint 2.1)
            """
            return model.soc[q] <= self.energy_cap

        def set_minimum_soc(model, q):
            """
            State of charge (MWh) can never be less than 0. (Constraint 2.2)
            """
            return model.soc[q] >= self.soc_min

        def set_first_soc_to_0(model):
            """
            State of charge (MWh)  at the first quarter must >=min. (Constraint 2.3)
            """
            return model.soc[1] >= self.soc_min
        
        def set_last_soc_to_0(model):
            """
            State of charge (MWh) at quarter 97 (i.e., first quarter of next day) must be >=min. (Constraint 2.4)
            """
            return model.soc[97] >= self.soc_min

        def soc_step_constraint(model, q):
            """
            The state of charge of each quarter equals the state if charge of the previous quarter plus charges minus discharges. (Constraint 2.5)
            """
            return model.soc[q + 1] == model.soc[q] + self.qBESSmax * (model.x_DAM_ch[q] - model.x_DAM_dis[q] + model.x_EC_ch1[q] - model.x_EC_dis1[q] + self.x_BM_ch[q-1] - self.x_BM_dis[q-1])

        def charge_cycle_limit(model):
            """
            Sum of all charges has to be below the daily limit. (Constraint 2.6)
            """
            return ((np.sum(self.x_BM_ch) + sum(model.x_DAM_ch[q] for q in model.Q)+ sum(model.x_EC_ch1[q] for q in model.Q) ) * self.qBESSmax <= self.volume_limit)

        def discharge_cycle_limit(model):
            """
            Sum of all discharges has to be below the daily limit. (Constraint 2.7)
            """
            return ((np.sum(self.x_BM_dis) + sum(model.x_DAM_dis[q] for q in model.Q)+ sum(model.x_EC_dis1[q] for q in model.Q) ) * self.qBESSmax <= self.volume_limit)
        # =============================================================================
        #Parity for charging and discharging positions over an hour
        # =============================================================================
        def cha_daa_quarters_1_2_parity(model, q):
            """
            Set DAM positions of quarter 1 and 2 of each hour equal. (Constraint 2.8)
            On the DAM Auction, positions in all 4 quarters of the hour have to be identical since trades are taken in hourly blocks. 
            """
            return model.x_DAM_ch[4 * q + 1] == model.x_DAM_ch[4 * q + 2]

        def cha_daa_quarters_2_3_parity(model, q):
            """
            Set DAM positions of quarter 2 and 3 of each hour equal. (Constraint 2.8)
            """
            return model.x_DAM_ch[4 * q + 2] == model.x_DAM_ch[4 * q + 3]

        def cha_daa_quarters_3_4_parity(model, q):
            """
            Set DAM positions of quarter 3 and 4 of each hour equal. (Constraint 2.8)
            """
            return model.x_DAM_ch[4 * q + 3] == model.x_DAM_ch[4 * q + 4]

        def dis_daa_quarters_1_2_parity(model, q):
            """
            Set DAM positions of quarter 1 and 2 of each hour equal. (Constraint 2.9)
            On the DAM Auction, positions in all 4 quarters of the hour have to be identical since trades are taken in hourly blocks. 
            """
            return model.x_DAM_dis[4 * q + 1] == model.x_DAM_dis[4 * q + 2]

        def dis_daa_quarters_2_3_parity(model, q):
            """
            Set DAM positions of quarter 2 and 3 of each hour equal. (Constraint 2.9)
            """
            return model.x_DAM_dis[4 * q + 2] == model.x_DAM_dis[4 * q + 3]

        def dis_daa_quarters_3_4_parity(model, q):
            """
            Set DAM positions of quarter 3 and 4 of each hour equal. (Constraint 2.9)
            """
            return model.x_DAM_dis[4 * q + 3] == model.x_DAM_dis[4 * q + 4]
        # =============================================================================
        # Charge and discharge limits
        def charge_rate_limit(model, q):
            """
             Sum of x_DAM_ch[q] and x_EC_ch1[q] and x_BM_ch[q-1] has to be less or equal to 1. (Constraint 2.14)
             """
            return model.x_DAM_ch[q] + model.x_EC_ch1[q]+ self.x_BM_ch[q-1] <= 1

        def discharge_rate_limit(model, q):
            """
             Sum of x_DAM_dis[q] and x_EC_dis1[q] and x_BM_dis[q-1] has to be less or equal to 1. (Constraint 2.15)
             """
            return model.x_DAM_dis[q] + model.x_EC_dis1[q] + self.x_BM_dis[q-1] <= 1
        
        # EC charging/discharging limits
        def charge_EC_limit(model,q):
            """Charging BESS from EC is possible only if the surplus is positive"""
            return model.x_EC_ch1[q]*self.qBESSmax<=self.S[q-1]
        
        def discharge_EC_limit(model,q):
            """Discharging BESS into EC is possible only if the demand is positive"""
            return model.x_EC_dis1[q]*self.qBESSmax<=self.D[q-1]
        # =============================================================================
        # Apply constraints on the model:
        model.set_maximum_soc = pyo.Constraint(model.Q_plus_1, rule=set_maximum_soc)
        model.set_minimum_soc = pyo.Constraint(model.Q_plus_1, rule=set_minimum_soc)
        model.set_first_soc_to_0 = pyo.Constraint(rule=set_first_soc_to_0)
        model.set_last_soc_to_0 = pyo.Constraint(rule=set_last_soc_to_0)
        model.soc_step_constraint = pyo.Constraint(model.Q, rule=soc_step_constraint)
        model.charge_cycle_limit = pyo.Constraint(rule=charge_cycle_limit)
        model.discharge_cycle_limit = pyo.Constraint(rule=discharge_cycle_limit)
        # Parity for charging and discharging for an hour
        model.x_DAM_ch_quarters_1_2_parity = pyo.Constraint(model.H, rule=cha_daa_quarters_1_2_parity)
        model.x_DAM_ch_quarters_2_3_parity = pyo.Constraint(model.H, rule=cha_daa_quarters_2_3_parity)
        model.x_DAM_ch_quarters_3_4_parity = pyo.Constraint(model.H, rule=cha_daa_quarters_3_4_parity)
        model.x_DAM_dis_quarters_1_2_parity = pyo.Constraint(model.H, rule=dis_daa_quarters_1_2_parity)
        model.x_DAM_dis_quarters_2_3_parity = pyo.Constraint(model.H, rule=dis_daa_quarters_2_3_parity)
        model.x_DAM_dis_quarters_3_4_parity = pyo.Constraint(model.H, rule=dis_daa_quarters_3_4_parity)

        # Charge and discharge limits
        model.charge_rate_limit = pyo.Constraint(model.Q, rule=charge_rate_limit)
        model.discharge_rate_limit = pyo.Constraint(model.Q, rule=discharge_rate_limit)
        # EC charging/discharging limits
        model.charge_EC_limit=pyo.Constraint(model.Q, rule=charge_EC_limit)
        model.discharge_EC_limit=pyo.Constraint(model.Q, rule=discharge_EC_limit)

        # Define objective function and solve the optimization problem.
        # Revenue from DAM
        RDAM_BESS=sum(self.pDAM[q-1] *self.qBESSmax *  (model.x_DAM_dis[q]  - model.x_DAM_ch[q]) for q in model.Q)      
        RDAM_PV=sum((self.S[q-1] - model.x_EC_ch1[q]*self.qBESSmax) * self.pDAM[q-1]-(self.D[q-1] - model.x_EC_dis1[q]*self.qBESSmax)* self.pDAM[q-1] for q in model.Q)
        
        RDAM=RDAM_BESS+RDAM_PV
        # Revenue normalized ->maximize
        R_norm = RDAM / self.payment_DAM #->maximize
        
        # GDI 
        GDI = sum(self.S[q-1] - model.x_EC_ch1[q]*self.qBESSmax for q in model.Q) / self.TotalGeneration #->minimize
       
        # The objective is to maximize revenue from DAM Auction trades over all possible charge-discharge schedules.
        model.obj = pyo.Objective(expr = self.w1 * R_norm - self.w2 * GDI, sense=pyo.maximize)

        solver = self.set_glpk_solver()
        solver.solve(model, timelimit=5)
        #print("Objective value:", pyo.value(model.obj))

        # Retrieve arrays of resulting optimal soc/charge/discharge schedules after the DA Auction:
        self.soc_DAM = [model.soc[q].value for q in range(1, len(self.pDAM) + 1)]
        self.x_DAM_ch = [model.x_DAM_ch[q].value for q in range(1, len(self.pDAM) + 1)]
        self.x_DAM_dis = [model.x_DAM_dis[q].value for q in range(1, len(self.pDAM) + 1)]
        self.x_EC_ch1 = [model.x_EC_ch1[q].value for q in range(1, len(self.pDAM) + 1)]
        self.x_EC_dis1 = [model.x_EC_dis1[q].value for q in range(1, len(self.pDAM) + 1)]

        # Convert lists to numpy arrays and replace None with 0
        self.x_DAM_ch = np.nan_to_num(np.array(self.x_DAM_ch, dtype=float))
        self.x_DAM_dis = np.nan_to_num(np.array(self.x_DAM_dis, dtype=float))
        self.x_EC_ch1 = np.nan_to_num(np.array(self.x_EC_ch1, dtype=float))
        self.x_EC_dis1 = np.nan_to_num(np.array(self.x_EC_dis1, dtype=float))

        # Calculate total charge and discharge from BM, DAM and EC
        self.x_ch1 = np.asarray(self.x_BM_ch) + self.x_DAM_ch + self.x_EC_ch1
        self.x_dis1 = np.asarray(self.x_BM_dis) + self.x_DAM_dis + self.x_EC_dis1

        # Calculate profit from Day-ahead auction trades:
        self.RDAM_BESS = sum([self.pDAM[q] * self.qBESSmax *  (self.x_DAM_dis[q]  - self.x_DAM_ch[q]) for q in range(len(self.pDAM))])
        self.RDAM_PV= sum((self.S[q] - self.x_EC_ch1[q]*self.qBESSmax) * self.pDAM[q]-(self.D[q] - self.x_EC_dis1[q]*self.qBESSmax)* self.pDAM[q] for q in range(len(self.pDAM)))       
        self.GDI = sum(self.S[q] - self.x_EC_ch1[q] for q in range(len(self.pDAM))) / self.TotalGeneration
        return(self.soc_DAM, self.x_DAM_ch, self.x_DAM_dis,self.x_EC_ch1, self.x_EC_dis1)

    def step3_optimize_idm(self):
        """
        Calculates optimal charge/discharge schedule on the intraday auction (ida) for a given 96-d ida_price_vector.

        Returns:
        - soc_IDM: Resulting state of charge schedule in MWH
        - x_IDM_ch: Resulting charges on IDM Auction as coeficients of the rated power
        - x_IDM_dis: Resulting discharges on IDM Auction as coeficients of the rated power
        - x_IDM_ch_close: Resulting charges on IDM Auction to close previous DA Auction positions as coeficients of the rated power
        - x_IDM_dis_close: Resulting discharge on IDM Auction to close previous DA Auction positions as coeficients of the rated power
        - profit_IDM: Profit from Day-ahead auction trades
        - x_EC_ch2: Charging from EC
        - x_EC_dis2: Discharging to EC
        - x_EC_ch: Total charging from EC
        - x_EC_dis: Total discharging to EC
        - x_T_ch: Combined charges from BM + DAM and ID Auction as coeficients of the rated power
        - x_T_dis: Combined discharges from BM + DAM and ID Auction as coeficients of the rated power
        """
        if self.update_proba:
            # retrieve the IDM probabilities closer to real time (1 hour ahead) if in the loop, otherwise they are already initialized 
            #self.BM_Down_proba=self.data['proba_Down'].values
            #self.BM_Up_proba=self.data['proba_Up'].values
            # Recompute the activation on BM
            self.x_BM_ch=np.array(self.x_aFRR_ch, dtype=float)*self.BM_Down_proba
            self.x_BM_dis=np.array(self.x_aFRR_dis, dtype=float)*self.BM_Up_proba
            # Recompute the total charge and discharge after DAM
            self.x_ch1 = np.asarray(self.x_BM_ch) + self.x_DAM_ch + self.x_EC_ch1
            self.x_dis1 = np.asarray(self.x_BM_dis) + self.x_DAM_dis + self.x_EC_dis1
        
        # Initialize pyomo model:
        model = pyo.ConcreteModel()

        # Set parameters:

        # Number of hours
        model.H = pyo.RangeSet(0, len(self.pIDM)/4-1)

        # Number of quarters
        model.Q = pyo.RangeSet(1, len(self.pIDM))

        # Number of quarters plus 1
        model.Q_plus_1 = pyo.RangeSet(1, len(self.pIDM)+1)

        # Initialize variables:

        # State of charge
        model.soc = pyo.Var(model.Q_plus_1, domain=pyo.Reals)

        # Charges on the intraday auction
        model.x_IDM_ch = pyo.Var(model.Q, domain=pyo.NonNegativeReals, bounds=(0, 1))

        # Discharges on the intraday auction
        model.x_IDM_dis = pyo.Var(model.Q, domain=pyo.NonNegativeReals, bounds=(0, 1))

        # Charges on the intraday auction to close previous positions from the day-ahead auction
        model.x_IDM_ch_close = pyo.Var(model.Q, domain=pyo.NonNegativeReals, bounds=(0, 1))

        # Charges on the intraday auction to close previous positions from the day-ahead auction
        model.x_IDM_dis_close = pyo.Var(model.Q, domain=pyo.NonNegativeReals, bounds=(0, 1))

        # Charges from EC
        model.x_EC_ch2 = pyo.Var(model.Q, domain=pyo.NonNegativeReals, bounds=(0, 1))

        # Discharges to EC
        model.x_EC_dis2 = pyo.Var(model.Q, domain=pyo.NonNegativeReals, bounds=(0, 1))

        # Define Constraints:

        def set_maximum_soc(model, q):
            """
            State of charge can never be higher than Energy Capacity. (Constraint 3.1)
            """
            return model.soc[q] <= self.energy_cap

        def set_minimum_soc(model, q):
            """
            State of charge can never be less than 0. (Constraint 3.2)
            """
            return model.soc[q] >= self.soc_min

        def set_first_soc_to_0(model):
            """
            State of charge at the first quarter must be >=min. (Constraint 3.3)
            """
            return model.soc[1] >= self.soc_min

        def set_last_soc_to_0(model):
            """
            State of charge at quarter 97 (i.e., first quarter of next day) must be >= min. (Constraint 3.4)
            """
            return model.soc[97] >= self.soc_min

        def soc_step_constraint(model, q):
            """
            The state of charge of each quarter equals the state if charge of the previous quarter plus charges minus discharges. (Constraint 3.5)
            """
            return model.soc[q+1] == model.soc[q] + self.qBESSmax * (model.x_EC_ch2[q] - model.x_EC_dis2[q]+model.x_IDM_ch[q] - model.x_IDM_dis[q] + model.x_IDM_ch_close[q] - model.x_IDM_dis_close[q] + self.x_ch1[q-1] - self.x_dis1[q-1])

        def charge_cycle_limit(model):
            """
            Sum of all charges has to be below the daily limit. (Constraint 3.6)
            """
            return ((np.sum(self.x_ch1) + sum(model.x_EC_ch2[q] for q in model.Q)+ sum(model.x_IDM_ch[q] for q in model.Q) - sum(model.x_IDM_dis_close[q] for q in model.Q)) * self.qBESSmax <= self.volume_limit)

        def discharge_cycle_limit(model):
            """
            Sum of all discharges has to be below the daily limit. (Constraint 3.7)
            """
            return ((np.sum(self.x_dis1) + sum(model.x_EC_dis2[q] for q in model.Q)+ sum(model.x_IDM_dis[q] for q in model.Q) - sum(model.x_IDM_ch_close[q] for q in model.Q)) * self.qBESSmax <= self.volume_limit)

        def cha_close_logic(model, q):
            """
            cha_ida_close can only close or reduce existing discharging DAM positions. They can only be placed, where dis_daa positions exist. (Constraint 3.8)
            """
            return model.x_IDM_ch_close[q] <= self.x_DAM_dis[q-1]

        def dis_close_logic(model, q):
            """
            dis_ida_close can only close or reduce existing charging DAM positions. They can only be placed, where cha_daa positions exist. (Constraint 3.9)
            """
            return model.x_IDM_dis_close[q] <= self.x_DAM_ch[q-1]

        def charge_rate_limit(model, q):
            """
             Sum of x_IDM_ch[q]+ x_EC_ch2[q] + previous charges has to be less or equal to 1. (Constraint 3.10)
             """
            return model.x_IDM_ch[q] +model.x_EC_ch2[q]+ self.x_ch1[q-1] <= 1

        def discharge_rate_limit(model, q):
            """
             Sum of dis_ida[q]+x_EC_dis2[q] + previous discharges has to be less or equal to 1. (Constraint 3.11)
             """
            return model.x_IDM_dis[q] + model.x_EC_dis2[q]+ self.x_dis1[q-1] <= 1
        
        # EC charging/discharging limits
        def charge_EC_limit(model,q):
            """Charging BESS from EC is possible only if the remaining surplus is positive"""
            return model.x_EC_ch2[q]*self.qBESSmax<=max(0, self.delta_S[q-1])

        def discharge_EC_limit(model,q):
            """Discharging BESS into EC is possible only if the remaining demand is positive"""
            return model.x_EC_dis2[q]*self.qBESSmax<=max(0, self.delta_D[q-1])

        # Apply constraints on the model:
        model.set_maximum_soc = pyo.Constraint(model.Q_plus_1, rule=set_maximum_soc)
        model.set_minimum_soc = pyo.Constraint(model.Q_plus_1, rule=set_minimum_soc)
        model.set_first_soc_to_0 = pyo.Constraint(rule=set_first_soc_to_0)
        model.set_last_soc_to_0 = pyo.Constraint(rule=set_last_soc_to_0)
        model.soc_step_constraint = pyo.Constraint(model.Q, rule=soc_step_constraint)
        model.charge_cycle_limit = pyo.Constraint(expr=charge_cycle_limit)
        model.discharge_cycle_limit = pyo.Constraint(expr=discharge_cycle_limit)
        model.cha_close_logic = pyo.Constraint(model.Q, rule=cha_close_logic)
        model.dis_close_logic = pyo.Constraint(model.Q, rule=dis_close_logic)
        model.charge_rate_limit = pyo.Constraint(model.Q, rule=charge_rate_limit)
        model.discharge_rate_limit = pyo.Constraint(model.Q, rule=discharge_rate_limit)
        # EC charging/discharging limits
        model.charge_EC_limit=pyo.Constraint(model.Q, rule=charge_EC_limit)
        model.discharge_EC_limit=pyo.Constraint(model.Q, rule=discharge_EC_limit)

        # Define objective function and solve the optimization problem
        # IDM revenue
        RIDM_BESS=sum(self.pIDM[q-1] * self.qBESSmax * (model.x_IDM_dis[q] + model.x_IDM_dis_close[q] - model.x_IDM_ch[q] - model.x_IDM_ch_close[q]) for q in model.Q)
        RIDM_PV=sum((self.delta_S[q-1] - model.x_EC_ch2[q]*self.qBESSmax) * self.pIDM[q-1]-(self.delta_D[q-1] - model.x_EC_dis2[q]*self.qBESSmax)* self.pIDM[q-1] for q in model.Q)
        RIDM=RIDM_BESS+RIDM_PV
        # Normalize the objective function to avoid numerical issues with large values.
        R_norm = RIDM / self.payment_IDM
        # GDI as surplus+deficit covered by BESS
        GDI = sum((self.delta_S[q-1]- model.x_EC_ch2[q]*self.qBESSmax)+ (self.delta_D[q-1]- model.x_EC_dis2[q]*self.qBESSmax) for q in model.Q) / self.TotalGeneration
        model.obj = pyo.Objective(expr = self.w1 * R_norm - self.w2 * GDI , sense=pyo.maximize)

        solver = self.set_glpk_solver()
        solver.solve(model, timelimit=5)
        #print("Objective value:", pyo.value(model.obj))

        # Retrieve arrays of resulting optimal soc/charge/discharge schedules after the IDM Auction:
        self.soc_idm = [model.soc[q].value for q in range(1, len(self.pIDM) + 1)]
        self.x_IDM_ch = [model.x_IDM_ch[q].value for q in range(1, len(self.pIDM) + 1)]
        self.x_IDM_dis = [model.x_IDM_dis[q].value for q in range(1, len(self.pIDM) + 1)]
        self.x_IDM_ch_close = [model.x_IDM_ch_close[q].value for q in range(1, len(self.pIDM) + 1)]
        self.x_IDM_dis_close = [model.x_IDM_dis_close[q].value for q in range(1, len(self.pIDM) + 1)]
        self.x_EC_ch2 = [model.x_EC_ch2[q].value for q in range(1, len(self.pIDM) + 1)]
        self.x_EC_dis2 = [model.x_EC_dis2[q].value for q in range(1, len(self.pIDM) + 1)]
        # Convert lists to numpy arrays and replace None with 0
        self.soc_idm = np.array(self.soc_idm, dtype=float)
        self.x_IDM_ch = np.array(self.x_IDM_ch, dtype=float)
        self.x_IDM_dis = np.array(self.x_IDM_dis, dtype=float)
        self.x_IDM_ch_close = np.array(self.x_IDM_ch_close, dtype=float)
        self.x_IDM_dis_close = np.array(self.x_IDM_dis_close, dtype=float)
        self.x_EC_ch2 = np.array(self.x_EC_ch2, dtype=float)
        self.x_EC_dis2 = np.array(self.x_EC_dis2, dtype=float)

        # Calculate profit from IDM auction trades:
        self.RIDM_BESS = sum(self.qBESSmax * self.pIDM[q] * (self.x_IDM_dis[q] + self.x_IDM_dis_close[q] - self.x_IDM_ch[q] - self.x_IDM_ch_close[q]) for q in range(len(self.pIDM)))
        self.RIDM_PV = sum((self.delta_S[q] - self.x_EC_ch2[q]*self.qBESSmax) * self.pIDM[q]-(self.delta_D[q] - self.x_EC_dis2[q]*self.qBESSmax)* self.pIDM[q] for q in range(len(self.pIDM)))
        
        # Profit from BM activation
        self.profit_BM_BESS=sum(self.qBESSmax * (self.pBMUp[q]*self.BM_Up_proba[q]*self.x_aFRR_dis[q] + self.pBMDown[q]*self.BM_Down_proba[q]*self.x_aFRR_ch[q]) for q in range(len(self.pBMDown)))
        
        return(self.soc_idm, self.x_IDM_ch, self.x_IDM_dis, self.x_IDM_ch_close, self.x_IDM_dis_close, self.x_EC_ch2, self.x_EC_dis2)

    def post_trading(self):
        # Calculate final benefits
        self.GDI = sum(self.S[q] - (self.x_EC_ch1[q] + self.x_EC_ch2[q])*self.qBESSmax for q in range(len(self.pIDM))) / self.TotalGeneration
        self.profit_aFRR_BM= self.profit_aFRR_BESS + self.profit_BM_BESS
        self.profit_market= self.profit_aFRR_BM+self.RDAM_BESS + self.RDAM_PV+self.RIDM_BESS+self.RIDM_PV
        # print(f'Profit market {self.profit_market}, GDI {self.GDI}, CS {self.CS}, Total Payment final {self.payf}, Total payment reduction {self.pay_reduction}, Total Benefit {self.benefit}')
        # print(f'Profit aFRR + BM {self.profit_aFRR_BM}')
        # Calculate charging and discharging quantities from the coefficients. 
        # For e.g. if x_DAM_ch=1, q_DAM_ch=1*10MW/4=2.5MWh

        q_DAM_ch=np.array(self.x_DAM_ch, dtype=float)*(self.qBESSmax)
        q_DAM_dis=np.array(self.x_DAM_dis, dtype=float)*(self.qBESSmax)
        q_A_ch=np.array(self.x_aFRR_ch, dtype=float)*(self.qBESSmax)
        q_A_dis=np.array(self.x_aFRR_dis, dtype=float)*(self.qBESSmax)
        q_BM_ch=np.array(self.x_BM_ch, dtype=float)*(self.qBESSmax)
        q_BM_dis=np.array(self.x_BM_dis, dtype=float)*(self.qBESSmax)
        q_IDM_ch=np.array(self.x_IDM_ch, dtype=float)*(self.qBESSmax)
        q_IDM_dis=np.array(self.x_IDM_dis, dtype=float)*(self.qBESSmax)
        q_IDM_ch_close=np.array(self.x_IDM_ch_close, dtype=float)*(self.qBESSmax)
        q_IDM_dis_close=np.array(self.x_IDM_dis_close, dtype=float)*(self.qBESSmax) 
        q_EC_ch=np.array(self.x_EC_ch1+self.x_EC_ch2, dtype=float)*(self.qBESSmax)
        q_EC_dis=np.array(self.x_EC_dis1+self.x_EC_dis2, dtype=float)*(self.qBESSmax)


        # calculate power from the coefficients
        p_BM_ch = np.array(self.x_BM_ch, dtype=float) * self.power_cap
        p_BM_dis = np.array(self.x_BM_dis, dtype=float) * self.power_cap
        p_DAM_ch = np.array(self.x_DAM_ch, dtype=float) * self.power_cap
        p_DAM_dis = np.array(self.x_DAM_dis, dtype=float) * self.power_cap
        p_A_ch = np.array(self.x_aFRR_ch, dtype=float) * self.power_cap
        p_A_dis = np.array(self.x_aFRR_dis, dtype=float) * self.power_cap   
        p_IDM_ch = np.array(self.x_IDM_ch, dtype=float) * self.power_cap
        p_IDM_dis = np.array(self.x_IDM_dis, dtype=float) * self.power_cap
        p_IDM_ch_close = np.array(self.x_IDM_ch_close, dtype=float) * self.power_cap
        p_IDM_dis_close = np.array(self.x_IDM_dis_close, dtype=float) * self.power_cap  
        p_EC_ch = (np.array(self.x_EC_ch1, dtype=float) + np.array(self.x_EC_ch2, dtype=float)) * self.power_cap
        p_EC_dis = (np.array(self.x_EC_dis1, dtype=float) + np.array(self.x_EC_dis2, dtype=float)) * self.power_cap


        # Creating a DataFrame from the output
        df = pd.DataFrame({
        'Price_aFRR_Down':self.pADown,
        'Price_aFRR_Up':self.pAUp,
        'Price_BM_Sur': self.pBMDown,
        'Price_BM_Def': self.pBMUp,
        'Price_DAM':self.pDAM,
        'Price_idm':self.pIDM,
        'Q_cha_aFRR': q_A_ch,
        'Q_dis_aFRR': q_A_dis,
        'Q_cha_BM': q_BM_ch,
        'Q_dis_BM': q_BM_dis,
        'Q_cha_dam': q_DAM_ch,
        'Q_dis_dam': q_DAM_dis,
        'Q_cha_idm': q_IDM_ch,
        'Q_dis_idm': q_IDM_dis,
        'Q_cha_idm_close': q_IDM_ch_close,
        'Q_dis_idm_close':q_IDM_dis_close,
        'Q_cha_EC': q_EC_ch,
        'Q_dis_EC': q_EC_dis,
        'P_cha_aFRR': p_A_ch,
        'P_dis_aFRR': p_A_dis,
        'P_cha_BM': p_BM_ch,
        'P_dis_BM': p_BM_dis,
        'P_cha_dam': p_DAM_ch,
        'P_dis_dam': p_DAM_dis,
        'P_cha_idm': p_IDM_ch,
        'P_dis_idm': p_IDM_dis,
        'P_cha_idm_close': p_IDM_ch_close,
        'P_dis_idm_close': p_IDM_dis_close,
        'P_cha_EC': p_EC_ch,
        'P_dis_EC': p_EC_dis,
        'SOC_aFRR':self.soc_aFRR,
        'SOC_dam':self.soc_DAM,
        'SOC_idm':self.soc_idm
        })
        df[['Q_dis_dam', 'Q_dis_idm', 'Q_dis_idm_close', 'Q_dis_aFRR', 'Q_dis_BM', 'Q_dis_EC']] *= -1
        df[['P_dis_dam', 'P_dis_idm', 'P_dis_idm_close', 'P_dis_aFRR', 'P_dis_BM', 'P_dis_EC']] *= -1
        df['P_cha']=df['P_cha_dam']+df[ 'P_cha_idm']+df['P_cha_BM']+df[ 'P_cha_idm_close']+df['P_cha_EC']
        df['P_dis']=df['P_dis_dam']+df['P_dis_idm']+df['P_dis_BM']+df['P_dis_idm_close']+df['P_dis_EC']
        df['P']=(df['P_cha']+df['P_dis'])

        # Total Q charged and discharged on markets
        df['Q_cha']=df['Q_cha_dam']+df[ 'Q_cha_idm']+df['Q_cha_BM']
        df['Q_dis']=df['Q_dis_dam']+df['Q_dis_idm']+df['Q_dis_BM']
        df['Q_cha_close']=df[ 'Q_cha_idm_close']
        df['Q_dis_close']=df['Q_dis_idm_close']
        df['SOC']=df['SOC_idm']
        df['QT_cha']=df['Q_cha']+df['Q_cha_EC']
        df['QT_dis']=df['Q_dis']+df['Q_dis_EC']

        # Calculate q buy and qsell
        df['Timestamp'] = self.data['Timestamp']
        df['Surplus'] = self.S
        df['Demand'] = self.D
        df['Surplus_updated'] = self.data['S_updated']
        df['Demand_updated'] = self.data['D_updated']
        df['delta_S'] = self.data['delta_S']
        df['delta_D'] = self.data['delta_D']
        df['Q_aFRR_buy'] = (df['Q_cha_aFRR']).clip(lower=0)
        df['Q_aFRR_sell'] = (df['Q_dis_aFRR']).clip(upper=0)
        df['Q_BM_buy'] = (df['Q_cha_BM']).clip(lower=0)
        df['Q_BM_sell'] = (df['Q_dis_BM']).clip(upper=0)
        df['Q_DAM']=df['Demand']-self.x_EC_dis1-(df['Surplus'] -self.x_EC_ch1) + self.x_DAM_ch -self.x_DAM_dis
        df['Q_IDM']=df['delta_D'] -self.x_EC_dis2-(df['delta_S'] -self.x_EC_ch2) + self.x_IDM_ch - self.x_IDM_dis
        df['Q_DAM_buy'] = df['Q_DAM'].clip(lower=0)
        df['Q_DAM_sell'] = df['Q_DAM'].clip(upper=0)
        df['Q_IDM_buy'] = df['Q_IDM'].clip(lower=0)
        df['Q_IDM_sell'] = df['Q_IDM'].clip(upper=0)
        df['Q_IDM_buy_close'] = self.x_IDM_ch_close
        df['Q_IDM_sell_close'] = self.x_IDM_dis_close

        return df
        