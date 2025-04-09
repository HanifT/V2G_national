import numpy as np
import pandas as pd
import pyomo.environ as pyomo
from pyomo.util.infeasible import log_infeasible_constraints
import logging
import pandas as pd
from collections import defaultdict
from wtp import calculate_wtp
logging.basicConfig(level=logging.INFO)
logging.getLogger("gurobipy").setLevel(logging.WARNING)
logging.getLogger("pyomo").setLevel(logging.WARNING)

class EVCSP_delay():

    def __init__(self, itinerary={}, itinerary_kwargs={}, inputs={}):
        self.sic = None  # ✅ Default value for infeasible cases
        if itinerary:
            self.inputs = self.ProcessItinerary(itinerary=itinerary['trips'], **itinerary_kwargs)
        else:
            self.inputs = inputs

        if self.inputs:
            self.Build()

    def ProcessItinerary(self,
                         itinerary,
                         home_charger_likelihood=1,
                         work_charger_likelihood=0.75,
                         destination_charger_likelihood=.1,
                         midnight_charging_prob=0.5,
                         consumption=782.928,  # J/meter
                         battery_capacity=60 * 3.6e6,  # J
                         initial_soc=.5,
                         final_soc=.5,
                         ad_hoc_charger_power=100e3,
                         home_charger_power=6.1e3,
                         work_charger_power=6.1e3,
                         destination_charger_power=100.1e3,
                         ac_dc_conversion_efficiency=.88,
                         max_soc=1,
                         min_soc=.2,
                         min_dwell_event_duration=15 * 60,
                         max_ad_hoc_event_duration=7.2e3,
                         min_ad_hoc_event_duration=5 * 60,
                         payment_penalty=60,
                         travel_penalty=1 * 60,
                         time_penalty= 60,
                         dwell_charge_time_penalty=0,
                         ad_hoc_charge_time_penalty=1,
                         tiles=7,
                         rng_seed=123,
                         residential_rate=None,
                         commercial_rate=None,
                         other_rate=0.42,  # Default value
                         home_penalty=0.0,  # No penalty at home
                         work_penalty=0.1,  # 20% of ad-hoc penalty at work
                         other_penalty=0.2,  # 40% of ad-hoc penalty at other locations
                         ad_hoc_penalty=1.0,  # Full penalty at ad-hoc chargers
                         **kwargs,
                         ):

        # Extract income and BEV range from itinerary
        income = itinerary['Income'].to_numpy()
        bev_range = (battery_capacity / consumption) * 0.000621371  #

        # Calculate WTP for each trip
        wtp = calculate_wtp(income, bev_range)

        # 1. Create a random seed if none provided
        if not rng_seed:
            rng_seed = np.random.randint(1e6)
        rng = np.random.default_rng(rng_seed)

        # 2. We only pick ONE random day for the entire itinerary:
        random_day = rng.integers(1, 366)  # random integer day in 1..365
        allow_midnight_charging = rng.random(len(itinerary)) < midnight_charging_prob
        # 3. Convert trip start times from HHMM to seconds-past-midnight
        start_times_hhmm = itinerary['STRTTIME'].to_numpy()
        end_times_hhmm = itinerary['ENDTIME'].to_numpy()
        start_times = (end_times_hhmm // 100)

        # 4. Driving / dwell durations
        durations = itinerary['TRVLCMIN'].copy().to_numpy()  # in minutes
        durations_sec = durations * 60
        dwell_times = itinerary['DWELTIME'].copy().to_numpy()
        dwell_times = np.where(dwell_times < 0, 1440, dwell_times)  # fix negative
        # dwell_times[dwell_times < 0] = dwell_times[dwell_times >= 0].mean()

        # 6. Expand trips/dwells by "tiles"
        trip_distances = itinerary['TRPMILES'].copy().to_numpy() * 1609.34
        trip_discharge = trip_distances * consumption
        dwell_times_sec = dwell_times
        dwells = np.tile(dwell_times_sec, tiles)  # dwell in seconds
        durations_sec = np.tile(durations_sec, tiles)

        # 7. Now build "absolute_start_times"
        n_trips = len(dwells)

        # Extract month and day from the itinerary
        months = itinerary['Month'].to_numpy()
        days = itinerary['Day'].to_numpy()

        def compute_doy(months, days, leap_year=False):
            # Days per month (for normal and leap years)
            days_per_month = np.array([31, 28 + leap_year, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
            # Compute the cumulative days at the start of each month
            start_of_month = np.insert(np.cumsum(days_per_month), 0, 0)[:-1]
            # Compute DOY
            doy = start_of_month[months - 1] + days

            return doy

        year = itinerary['Year'].iloc[0] if 'Year' in itinerary.columns else 2024  # Default to 2024
        is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
        # Convert month/day to day of the year (DOY)
        doy = compute_doy(months, days, leap_year=is_leap)
        # Calculate the absolute start time in seconds
        absolute_start_times = doy * 24 + start_times  # DOY to seconds + HHMM in seconds
        d_e_max_hours = dwell_times.astype(int) / 60
        absolute_start_hours = absolute_start_times.astype(int)

        # 8. Location type arrays
        location_types = np.tile(itinerary['WHYTRP1S'].to_numpy(), tiles)
        is_home = location_types == 1
        is_work = location_types == 10
        is_other = ((~is_home) & (~is_work))
        # 9. Charger assignment (like before)
        generator = np.random.default_rng(seed=rng_seed)
        destination_charger_power_array = np.zeros(n_trips)
        destination_charger_power_array[is_home] = home_charger_power
        destination_charger_power_array[is_work] = work_charger_power
        destination_charger_power_array[is_other] = destination_charger_power
        home_charger_selection = generator.random(is_home.sum()) <= home_charger_likelihood
        work_charger_selection = generator.random(is_work.sum()) <= work_charger_likelihood
        destination_charger_selection = generator.random(is_other.sum()) <= destination_charger_likelihood

        destination_charger_power_array[is_home] = \
            (np.where(home_charger_selection, destination_charger_power_array[is_home], 0))

        destination_charger_power_array[is_work] =\
            (np.where(work_charger_selection, destination_charger_power_array[is_work], 0))

        destination_charger_power_array[is_other] = \
            ( np.where(destination_charger_selection, destination_charger_power_array[is_other], 0))

        # 10. Build inputs
        inputs = {}
        inputs['allow_midnight_charging'] = allow_midnight_charging
        inputs['end_times_hhmm'] = end_times_hhmm
        inputs['residential_rate'] = residential_rate
        inputs['commercial_rate'] = commercial_rate
        inputs['other_rate'] = other_rate
        inputs['n_e'] = n_trips
        inputs['s_i'] = initial_soc * battery_capacity
        inputs['s_f'] = final_soc * battery_capacity
        inputs['s_ub'] = max_soc * battery_capacity
        inputs['s_lb'] = min_soc * battery_capacity
        # Penalties
        inputs['c_db'] = np.ones(n_trips) * (payment_penalty + time_penalty) * is_other
        inputs['c_dd'] = np.ones(n_trips) * dwell_charge_time_penalty
        inputs['c_ab'] = np.ones(n_trips) * (travel_penalty + payment_penalty + time_penalty)
        inputs['c_at'] = np.ones(n_trips) * travel_penalty
        inputs['c_ad'] = np.ones(n_trips) * ad_hoc_charge_time_penalty
        # Charger data
        inputs['r_d'] = destination_charger_power_array
        inputs['r_d_h'] = home_charger_power
        inputs['r_d_w'] = work_charger_power
        inputs['r_d_o'] = destination_charger_power
        inputs['r_a'] = ad_hoc_charger_power
        # Time bounds for each event
        inputs['d_e_max'] = np.ones(n_trips) * d_e_max_hours
        inputs['d_e_min'] = np.ones(n_trips) * min_dwell_event_duration
        inputs['a_e_min'] = np.ones(n_trips) * min_ad_hoc_event_duration
        inputs['a_e_max'] = np.ones(n_trips) * max_ad_hoc_event_duration
        # For the trip consumption
        inputs['d'] = trip_discharge
        inputs['b_c'] = battery_capacity
        inputs['l_i'] = trip_distances.sum()
        # location flags
        inputs['is_home'] = is_home
        inputs['is_work'] = is_work
        inputs['is_other'] = is_other
        # The final absolute start times
        inputs['absolute_start_times'] = absolute_start_hours
        inputs['home_penalty'] = home_penalty
        inputs['work_penalty'] = work_penalty
        inputs['other_penalty'] = other_penalty
        inputs['ad_hoc_penalty'] = ad_hoc_penalty
        inputs['ad_hoc_travel_time_loss'] = travel_penalty  # 15 minutes at the start and 15 at the end
        inputs['wtp'] = wtp * 60
        # Store valid charging hours for each event
        # Cap hours to 8759 (end of the year)
        inputs['valid_hours'] = {
            e: [
                m for m in range(
                    int(inputs['absolute_start_times'][e]),
                    int((inputs['absolute_start_times'][e] + inputs['d_e_max'][e]))
                )
                if 0 <= m <= 8759
            ]
            for e in range(n_trips)
        }

        if isinstance(inputs['residential_rate'], list) and len(inputs['residential_rate']) == 8760:
            mean_res_rate = np.mean(inputs['residential_rate'])
            inputs['residential_rate'].append(mean_res_rate)  # Add hour 8760

        if isinstance(inputs['commercial_rate'], list) and len(inputs['commercial_rate']) == 8760:
            mean_com_rate = np.mean(inputs['commercial_rate'])
            inputs['commercial_rate'].append(mean_com_rate)  # Add hour 8760

        # If needed, transform the rate arrays from dict to list
        if isinstance(inputs['residential_rate'], dict) and "rate" in inputs['residential_rate']:
            inputs['residential_rate'] = [inputs['residential_rate']["rate"][str(hour)] for hour in range(8761)]

        if isinstance(inputs['commercial_rate'], dict) and "rate" in inputs['commercial_rate']:
            inputs['commercial_rate'] = [inputs['commercial_rate']["rate"][str(hour)] for hour in range(8761)]

        return inputs

    def Solve(self, solver_kwargs={}):
        solver = pyomo.SolverFactory(**solver_kwargs)
        # Add MIP gap for groubi (set to 5%)
        solver.options['MIPGap'] = 0.35  # 35% MIP gap
        solver.options['Threads'] = 24
        solver.options['Presolve'] = 2
        solver.options['Heuristics'] = 0.5  # Enable solver's internal heuristic
        solver.options["OutputFlag"] = 0
        # solver.options["MIPFocus"] = 1  # Focus on finding feasible solutions faster
        # solver.options["Method"] = 2  # Barrier method for faster relaxation
        # solver.options["NodeMethod"] = 2  # Parallelized branch-and-bound
        # solver.options['Cuts'] = 3 # or 3
        # solver.options["LazyConstraints"] = 1  # Enable lazy constraints (can speed up MIP)

        # Write the LP file for debugging
        self.model.write("model.lp", io_options={"symbolic_solver_labels": True})
        res = solver.solve(self.model, tee=False)
        self.solver_status = res.solver.status
        self.solver_termination_condition = res.solver.termination_condition

        # ✅ Check if the solver found a feasible solution
        if self.solver_termination_condition in ["optimal", "locally optimal", "feasible"]:
            self.Solution()
            self.Compute()
        else:
            print(f"WARNING: Solver returned infeasible solution for {self.inputs.get('itinerary_id', 'unknown')}")

    def Build(self):

        # Pulling the keys from the inputs dict
        keys = self.inputs.keys()
        # Initializing the model as a concrete model
        self.model = pyomo.ConcreteModel(name="EVCSP_Model")
        # Adding variables
        self.Variables()
        # Upper-level objective (inconvenience)
        self.UpperObjective()  # New
        # Lower-level KKT conditions for cost minimization
        # self.LowerKKT()
        self.LowerKKT_new()
        # Bounds constraints
        self.Bounds()
        # Unit commitment constraints
        # self.Unit_Commitment()

    def Variables(self):
        # **Define event set**
        self.model.E = pyomo.Set(initialize=range(self.inputs['n_e']))
        # **Minute-based resolution**
        EM = [(e, h) for e in range(self.inputs['n_e']) for h in self.inputs['valid_hours'][e]]
        self.model.EM = pyomo.Set(initialize=EM, dimen=2)

        # **Charging rate decision variables**
        self.model.x_d = pyomo.Var(self.model.EM, domain=pyomo.NonNegativeReals)
        self.model.x_a = pyomo.Var(self.model.EM, domain=pyomo.NonNegativeReals, bounds=(0, self.inputs['r_a']))

        # **Emergency Charging
        self.model.x_a_nec = pyomo.Var(self.model.E, domain=pyomo.NonNegativeReals, bounds=(0, self.inputs['r_a']))

        # **Charging duration decision variable**
        self.model.tdu_e = pyomo.Var(self.model.E, domain=pyomo.NonNegativeReals)
        self.model.tau_e = pyomo.Var(self.model.E, domain=pyomo.NonNegativeReals)

        self.model.u_db = pyomo.Var(self.model.E, domain=pyomo.Boolean)
        self.model.u_ab = pyomo.Var(self.model.E, domain=pyomo.Boolean)

    def UpperObjective(self):

        destination_charge_event = sum(
            self.inputs['c_db'][e] * self.model.u_db[e] for e in self.model.E)

        destination_charge_duration = sum(
            self.inputs['c_dd'][e] * self.model.tdu_e[e] * self.inputs['wtp'][e] for e in self.model.E) \
            if any(self.inputs['c_dd']) else 0

        ad_hoc_charge_event = sum(
            self.inputs['c_ab'][e] * self.model.u_ab[e] for e in self.model.E)

        ad_hoc_charge_duration = sum(
            self.inputs['c_ad'][e] * self.model.tau_e[e] * self.inputs['wtp'][e] for e in self.model.E)  \
            if any(self.inputs['c_ad']) else 0

        self.model.objective = pyomo.Objective(
            expr=(
                         destination_charge_event +
                         destination_charge_duration +
                         ad_hoc_charge_event +
                         ad_hoc_charge_duration
            ),
            sense=pyomo.minimize
        )

        self.model.objective.expr += sum(100 * 1e9 * self.model.x_a_nec[e] for e in self.model.E)

    # def LowerKKT(self):
    #     model = self.model
    #
    #     # 1) Dual Variables for Charging Decisions
    #     model.lambda_d_min = pyomo.Var(model.EM, domain=pyomo.NonNegativeReals)
    #     model.lambda_d_max = pyomo.Var(model.EM, domain=pyomo.NonNegativeReals)
    #     model.lambda_a_min = pyomo.Var(model.EM, domain=pyomo.NonNegativeReals)
    #     model.lambda_a_max = pyomo.Var(model.EM, domain=pyomo.NonNegativeReals)
    #
    #     # 2) Dual Variables for Duration Constraints
    #     model.mu_tau_min = pyomo.Var(model.E, domain=pyomo.NonNegativeReals)
    #     model.mu_tau_max = pyomo.Var(model.E, domain=pyomo.NonNegativeReals)
    #     model.mu_tdu_min = pyomo.Var(model.E, domain=pyomo.NonNegativeReals)
    #     model.mu_tdu_max = pyomo.Var(model.E, domain=pyomo.NonNegativeReals)
    #
    #     # 3) Auxiliary Variables for Linearization
    #     model.z_d = pyomo.Var(model.EM, domain=pyomo.NonNegativeReals)
    #     model.z_a = pyomo.Var(model.EM, domain=pyomo.NonNegativeReals)
    #
    #     # 4) Stationarity Conditions for Destination Charging (x_d)
    #     def stationarity_xd_rule(m, e, h):
    #         """ Ensure correct stationarity for destination charging (x_d) """
    #         hour_index = max(0, min(h, 8759))
    #
    #         # Select rate per hour
    #         cost_rate_home = self.inputs['residential_rate'][hour_index] if isinstance(self.inputs['residential_rate'], list) else self.inputs['residential_rate']
    #         cost_rate_work = self.inputs['commercial_rate'][hour_index] if isinstance(self.inputs['commercial_rate'], list) else self.inputs['commercial_rate']
    #         cost_rate_other = self.inputs['other_rate'][hour_index] if isinstance(self.inputs['other_rate'], list) else self.inputs['other_rate']
    #
    #         # Compute cost rate for each event and hour
    #         cost_rate = (
    #                 cost_rate_home * self.inputs['is_home'][e] +
    #                 cost_rate_work * self.inputs['is_work'][e] +
    #                 cost_rate_other * self.inputs['is_other'][e]
    #         )
    #
    #         # Stationarity condition for x_d
    #         return cost_rate - m.lambda_d_min[e, h] + m.lambda_d_max[e, h] == 0
    #
    #     model.stationarity_xd = pyomo.Constraint(model.EM, rule=stationarity_xd_rule)
    #
    #     # 5) Stationarity Conditions for Ad-Hoc Charging (x_a)
    #     def stationarity_xa_rule(m, e, h):
    #         cost_rate = self.inputs['other_rate']
    #         return cost_rate - m.lambda_a_min[e, h] + m.lambda_a_max[e, h] == 0
    #
    #     model.stationarity_xa = pyomo.Constraint(model.EM, rule=stationarity_xa_rule)
    #
    #     # 6) Primal Feasibility for Charging Speed Constraints
    #     def xd_min_rule(m, e, h):
    #         return m.x_d[e, h] >= 0
    #
    #     def xd_max_rule(m, e, h):
    #         max_power = (
    #             self.inputs['r_d_h'] if self.inputs['is_home'][e] else
    #             self.inputs['r_d_w'] if self.inputs['is_work'][e] else
    #             self.inputs['r_d_o'] if self.inputs['is_other'][e] else 0
    #         )
    #         return m.x_d[e, h] <= max_power * m.z_d[e, h]
    #
    #     model.xd_min_con = pyomo.Constraint(model.EM, rule=xd_min_rule)
    #     model.xd_max_con = pyomo.Constraint(model.EM, rule=xd_max_rule)
    #
    #     def xa_min_rule(m, e, h):
    #         return m.x_a[e, h] >= 0
    #
    #     def xa_max_rule(m, e, h):
    #         return m.x_a[e, h] <= self.inputs['r_a'] * m.z_a[e, h]
    #
    #     model.xa_min_con = pyomo.Constraint(model.EM, rule=xa_min_rule)
    #     model.xa_max_con = pyomo.Constraint(model.EM, rule=xa_max_rule)
    #
    #     # 7) Link Auxiliary Variables to Binary Decisions
    #     M = 1e9
    #
    #     def link_zd_rule_1(m, e, h):
    #         """ Link z_d to binary decision u_db with Big-M """
    #         return m.x_d[e, h] <= M * m.u_db[e]
    #
    #     def link_zd_rule_2(m, e, h):
    #         """ Enforce z_d to be 0 if u_db is 0 """
    #         return m.z_d[e, h] <= m.u_db[e]
    #
    #     def link_za_rule_1(m, e, h):
    #         """ Link z_a to binary decision u_ab with Big-M """
    #         return m.x_a[e, h] <= M * m.u_ab[e]
    #
    #     def link_za_rule_2(m, e, h):
    #         """ Enforce z_a to be 0 if u_ab is 0 """
    #         return m.z_a[e, h] <= m.u_ab[e]
    #     #
    #     model.link_zd_con_1 = pyomo.Constraint(model.EM, rule=link_zd_rule_1)
    #     model.link_zd_con_2 = pyomo.Constraint(model.EM, rule=link_zd_rule_2)
    #     model.link_za_con_1 = pyomo.Constraint(model.EM, rule=link_za_rule_1)
    #     model.link_za_con_2 = pyomo.Constraint(model.EM, rule=link_za_rule_2)
    #
    #     # 8) Primal Feasibility for Duration Constraints
    #     def tdu_min_rule(m, e):
    #         return m.tdu_e[e] >= 0
    #
    #     def tdu_max_rule(m, e):
    #         return m.tdu_e[e] <= self.inputs['d_e_max'][e]
    #
    #     model.tdu_min_con = pyomo.Constraint(model.E, rule=tdu_min_rule)
    #     model.tdu_max_con = pyomo.Constraint(model.E, rule=tdu_max_rule)
    #
    #     def tau_min_rule(m, e):
    #         return m.tau_e[e] >= 0
    #
    #     def tau_max_rule(m, e):
    #         return m.tau_e[e] <= self.inputs['a_e_max'][e]
    #
    #     model.tau_min_con = pyomo.Constraint(model.E, rule=tau_min_rule)
    #     model.tau_max_con = pyomo.Constraint(model.E, rule=tau_max_rule)
    #
    #     # 9) Big-M Linearization for Complementary Slackness
    #     M = 150  #Big-M value (large positive number)
    #
    #     def compl_slack_tdu_min_rule(m, e):
    #         return m.mu_tdu_min[e] <= M * (1 - m.u_db[e])
    #
    #     def compl_slack_tdu_max_rule(m, e):
    #         return m.mu_tdu_max[e] <= M * m.u_db[e]
    #
    #     model.compl_slack_tdu_min = pyomo.Constraint(model.E, rule=compl_slack_tdu_min_rule)
    #     model.compl_slack_tdu_max = pyomo.Constraint(model.E, rule=compl_slack_tdu_max_rule)
    #
    #     def compl_slack_tau_min_rule(m, e):
    #         return m.mu_tau_min[e] <= M * (1 - m.u_ab[e])
    #
    #     def compl_slack_tau_max_rule(m, e):
    #         return m.mu_tau_max[e] <= M * m.u_ab[e]
    #
    #     model.compl_slack_tau_min = pyomo.Constraint(model.E, rule=compl_slack_tau_min_rule)
    #     model.compl_slack_tau_max = pyomo.Constraint(model.E, rule=compl_slack_tau_max_rule)
    def LowerKKT_new(self):


        model = self.model
        E = model.E  # set of events
        EM = model.EM  # set of (e,h) pairs where charging is allowed

        # -------------------------------------------------------
        # 1) Define a helper to get the cost rate for destination vs. ad-hoc
        # -------------------------------------------------------
        def cost_rate_destination(e, h):
            """Return the electricity price for destination/home/work charging."""
            hour_index = max(0, min(h, 8759))
            # Distinguish home vs. work vs. other
            if self.inputs['is_home'][e]:
                # If you have a time-varying residential rate:
                if isinstance(self.inputs['residential_rate'], list):
                    return self.inputs['residential_rate'][hour_index]
                else:
                    return self.inputs['residential_rate']
            elif self.inputs['is_work'][e]:
                if isinstance(self.inputs['commercial_rate'], list):
                    return self.inputs['commercial_rate'][hour_index]
                else:
                    return self.inputs['commercial_rate']
            else:
                # 'Other' location => possibly a flat or time-varying rate
                return self.inputs['other_rate']

        def cost_rate_adhoc(e, h):
            return self.inputs['other_rate']

        # -------------------------------------------------------
        # 2) Define Dual Variables for Each Constraint
        # -------------------------------------------------------
        # For x_d:  0 <= x_d[e,h] <= r_d(e),   sum_h x_d[e,h] <= tdu_e[e]* r_d(e)
        model.lambda_d_min = pyomo.Var(EM, domain=pyomo.NonNegativeReals)  # for x_d[e,h] >= 0
        model.lambda_d_max = pyomo.Var(EM, domain=pyomo.NonNegativeReals)  # for x_d[e,h] <= r_d(e)
        model.mu_d = pyomo.Var(E, domain=pyomo.NonNegativeReals)  # for sum_h x_d <= tdu_e[e]*r_d(e)

        # For x_a:  0 <= x_a[e,h] <= r_a,     sum_h x_a[e,h] <= tau_e[e]* r_a
        model.lambda_a_min = pyomo.Var(EM, domain=pyomo.NonNegativeReals)  # for x_a[e,h] >= 0
        model.lambda_a_max = pyomo.Var(EM, domain=pyomo.NonNegativeReals)  # for x_a[e,h] <= r_a
        model.mu_a = pyomo.Var(E, domain=pyomo.NonNegativeReals)  # for sum_h x_a <= tau_e[e]*r_a

        # -------------------------------------------------------
        # 3) Primal Feasibility Constraints
        # -------------------------------------------------------
        # x_d[e,h] >= 0, x_d[e,h] <= r_d(e)
        # We'll keep them as normal constraints so that x_d doesn't become negative, etc.

        def xd_max_rule(m, e, h):
            # r_dest = (self.inputs['r_d_h'] if self.inputs['is_home'][e]
            #           else self.inputs['r_d_w'] if self.inputs['is_work'][e]
            #           else self.inputs['r_d_o'] if self.inputs['is_other'][e]
            #           else 0)
            r_dest = self.inputs['r_d'][e]  # event-specific destination charging power
            return m.x_d[e, h] <= r_dest * self.model.u_db[e]

        model.xd_max_con = pyomo.Constraint(EM, rule=xd_max_rule)

        # sum_h x_d[e,h] <= tdu_e[e]* r_dest
        def sum_xd_rule(m, e):
            # r_dest = (self.inputs['r_d_h'] if self.inputs['is_home'][e]
            #           else self.inputs['r_d_w'] if self.inputs['is_work'][e]
            #           else self.inputs['r_d_o'] if self.inputs['is_other'][e]
            #           else 0)
            r_dest = self.inputs['r_d'][e]  # event-specific destination charging power
            return m.tdu_e[e] * r_dest - sum(m.x_d[e, h] for h in self.inputs['valid_hours'][e]) >= 0
        model.sum_xd_con = pyomo.Constraint(E, rule=sum_xd_rule)

        # Similarly for x_a[e,h]:

        def xa_max_rule(m, e, h):
            return m.x_a[e, h] <= self.inputs['r_a'] * self.model.u_ab[e]

        model.xa_max_con = pyomo.Constraint(EM, rule=xa_max_rule)

        def sum_xa_rule(m, e):
            return m.tau_e[e] * self.inputs['r_a'] - sum(m.x_a[e, h] for h in self.inputs['valid_hours'][e]) >= 0

        model.sum_xa_con = pyomo.Constraint(E, rule=sum_xa_rule)

        # -------------------------------------------------------
        # 4) Stationarity Conditions
        # -------------------------------------------------------
        # The follower's cost is:
        #  Cost = sum_{(e,h)} [ cost_rate_destination(e,h)* x_d[e,h]  +  cost_rate_adhoc(e,h)* x_a[e,h] ].
        #
        # We'll write the constraints ensuring partial derivatives (w.r.t x_d and x_a) are zero,
        # accounting for the Lagrange terms from primal constraints.
        #
        # If the constraint is "sum x_d(e,h) - tdu_e[e]*r_d(e) <= 0" with multiplier mu_d[e],
        # the partial derivative w.r.t x_d[e,h] is + mu_d[e].
        #
        # => Stationarity for x_d[e,h]:
        #    cost_rate_destination(e,h)
        #    - lambda_d_min[e,h]
        #    + lambda_d_max[e,h]
        #    + mu_d[e]
        #    = 0    (for an optimum in a linear program, if x_d[e,h] is "in the interior" or if we do full MPEC)
        #
        # We'll implement it as an equality for all (e,h).
        #
        def stationarity_xd_rule(m, e, h):
            return (
                    cost_rate_destination(e, h)
                    - m.lambda_d_min[e, h]
                    + m.lambda_d_max[e, h]
                    - m.mu_d[e]
            ) == 0

        model.stationarity_xd = pyomo.Constraint(EM, rule=stationarity_xd_rule)

        # Stationarity for x_a[e,h]:
        #   cost_rate_adhoc(e,h)
        #   - lambda_a_min[e,h]
        #   + lambda_a_max[e,h]
        #   + mu_a[e]
        #   = 0
        #
        def stationarity_xa_rule(m, e, h):
            return (
                    cost_rate_adhoc(e, h)
                    - m.lambda_a_min[e, h]
                    + m.lambda_a_max[e, h]
                    - m.mu_a[e]
            ) == 0

        model.stationarity_xa = pyomo.Constraint(EM, rule=stationarity_xa_rule)

        # -------------------------------------------------------
        # 5) Complementary Slackness
        # -------------------------------------------------------
        # For constraints like x_d[e,h] >= 0, we have a multiplier lambda_d_min[e,h].
        # => lambda_d_min[e,h] * x_d[e,h] == 0
        #
        # For x_d[e,h] <= r_dest => lambda_d_max[e,h] * (r_dest - x_d[e,h]) == 0
        #
        # For sum_x_d[e,h] <= tdu_e[e]* r_dest => mu_d[e] * (tdu_e[e]*r_dest - sum(x_d[e,h])) == 0
        #
        # And similarly for x_a.
        #
        def compl_slack_xd_min_rule(m, e, h):
            return m.lambda_d_min[e, h] * m.x_d[e, h] <= 1e-3

        def compl_slack_xd_max_rule(m, e, h):
            # r_dest = (self.inputs['r_d_h'] if self.inputs['is_home'][e]
            #           else self.inputs['r_d_w'] if self.inputs['is_work'][e]
            #           else self.inputs['r_d_o'] if self.inputs['is_other'][e]
            #           else 0)
            r_dest = self.inputs['r_d'][e]  # event-specific destination charging power
            return m.lambda_d_max[e, h] * (r_dest - m.x_d[e, h]) <= 1e-3

        def compl_slack_sumxd_rule(m, e):
            # r_dest = (self.inputs['r_d_h'] if self.inputs['is_home'][e]
            #           else self.inputs['r_d_w'] if self.inputs['is_work'][e]
            #           else self.inputs['r_d_o'] if self.inputs['is_other'][e]
            #           else 0)
            r_dest = self.inputs['r_d'][e]  # event-specific destination charging power
            lhs = r_dest * m.tdu_e[e] - sum(m.x_d[e, h] for h in self.inputs['valid_hours'][e])
            return m.mu_d[e] * lhs <= 1e-3

        model.compl_slack_xd_min = pyomo.Constraint(EM, rule=compl_slack_xd_min_rule)
        model.compl_slack_xd_max = pyomo.Constraint(EM, rule=compl_slack_xd_max_rule)
        model.compl_slack_sumxd = pyomo.Constraint(E, rule=compl_slack_sumxd_rule)

        # Now for x_a:
        def compl_slack_xa_min_rule(m, e, h):
            return m.lambda_a_min[e, h] * m.x_a[e, h] <= 1e-3

        def compl_slack_xa_max_rule(m, e, h):
            return m.lambda_a_max[e, h] * (self.inputs['r_a'] - m.x_a[e, h]) <= 1e-3

        def compl_slack_sumxa_rule(m, e):
            lhs = self.inputs['r_a'] * m.tau_e[e] - sum(m.x_a[e, h] for h in self.inputs['valid_hours'][e])
            return m.mu_a[e] * lhs <= 1e-3

        model.compl_slack_xa_min = pyomo.Constraint(EM, rule=compl_slack_xa_min_rule)
        model.compl_slack_xa_max = pyomo.Constraint(EM, rule=compl_slack_xa_max_rule)
        model.compl_slack_sumxa = pyomo.Constraint(E, rule=compl_slack_sumxa_rule)

    def Bounds(self):
        model = self.model

        # Create a ConstraintList to store SOC constraints
        model.bounds = pyomo.ConstraintList()

        # Define SOC Variables (still in Joules)
        model.ess_state = pyomo.Var(
            model.E, domain=pyomo.NonNegativeReals,
            bounds=(self.inputs['s_lb'], self.inputs['s_ub'])
        )
        model.soc_after_trip = pyomo.Var(
            model.E, domain=pyomo.NonNegativeReals,
            bounds=(self.inputs['s_lb'], self.inputs['s_ub'])
        )

        # Upper-Level Charging Duration Constraints
        for e in model.E:
            if e == self.inputs['n_e'] - 1:
                # Last event: No upper bounds on charging duration
                # print(f"Relaxing bounds for last event: {e}")
                continue  # Skip bounds for the last event

                # Apply bounds for all other events
            model.bounds.add(model.tau_e[e] <= self.inputs['a_e_max'][e])  # Ensure charging fits within max dwell
            model.bounds.add(model.tdu_e[e] <= self.inputs['d_e_max'][e])  # Max destination charging time

            # Adjust Charging Window for Ad-Hoc Travel Time Loss (minutes)
            travel_time_loss = self.inputs['ad_hoc_travel_time_loss']
            adjusted_time = max(self.inputs['d_e_max'][e] - 2 * travel_time_loss, 0)
            model.bounds.add(model.tau_e[e] <= adjusted_time)

        # First handle the SOC after each trip (in Joules)
        for e in model.E:

            if e == 0:
                model.bounds.add(
                    model.soc_after_trip[e] == self.inputs['s_i'] - self.inputs['d'][e]
                )
            else:
                model.bounds.add(
                    model.soc_after_trip[e] == model.ess_state[e - 1] - self.inputs['d'][e]
                )

        # Now link ess_state[e] to soc_after_trip[e] + sum of charging (Joules)
        for e in model.E:
            # Summation over all valid hours for event e
            model.bounds.add(
                model.ess_state[e] ==
                model.soc_after_trip[e] +
                sum(model.x_d[e, h] + model.x_a[e, h] for h in self.inputs['valid_hours'][e]) + model.x_a_nec[e]
            )

        # Final SOC Must Meet Required Final SOC (still in Joules)
        model.bounds.add(
            model.ess_state[self.inputs['n_e'] - 1] >= self.inputs['s_f']
        )

    def Unit_Commitment(self):
        model = self.model
        # Create a ConstraintList for unit commitment constraints
        model.unit_commitment = pyomo.ConstraintList()

        # 1️⃣ Charging Only Within Valid Hours
        def charging_within_window_rule_d(model, e, h):
            """Destination charging (x_d) must only happen within valid hours."""
            if h not in self.inputs['valid_hours'][e]:
                return model.x_d[e, h] == 0
            return pyomo.Constraint.Skip

        # model.no_charge_outside_window_d = pyomo.Constraint(model.EM, rule=charging_within_window_rule_d)

        def charging_within_window_rule_a(model, e, h):
            """Ad-Hoc charging (x_a) must only happen within valid hours."""
            if h not in self.inputs['valid_hours'][e]:
                return model.x_a[e, h] == 0
            return pyomo.Constraint.Skip

        # model.no_charge_outside_window_a = pyomo.Constraint(model.EM, rule=charging_within_window_rule_a)

        # 2️⃣ Link Duration Variables (`tdu_e`, `tau_e`) to Charging Decisions
        for e in model.E:
            model.unit_commitment.add(
                self.inputs['d_e_min'][e] * self.model.u_db[e] <= model.tdu_e[e]  # Min duration for destination
            )
            model.unit_commitment.add(
                self.inputs['d_e_max'][e] * self.model.u_db[e] >= model.tdu_e[e]  # Max duration for destination
            )

            model.unit_commitment.add(
                self.inputs['a_e_min'][e] * self.model.u_ab[e] <= model.tau_e[e]  # Min duration for ad-hoc
            )
            model.unit_commitment.add(
                self.inputs['a_e_max'][e] * self.model.u_ab[e] >= model.tau_e[e]  # Max duration for ad-hoc
            )

        # 3️⃣ Duration Consistency: Link Duration to Charging Rates
        def destination_duration_limit(model, e):
            """Destination charging duration limit linked to `tdu_e`."""
            return sum(
                model.x_d[e, h] / (
                    self.inputs['r_d_h'] if self.inputs['is_home'][e] else
                    self.inputs['r_d_w'] if self.inputs['is_work'][e] else
                    self.inputs['r_d_o'] if self.inputs['is_other'][e] else 0
                )
                for h in self.inputs['valid_hours'][e]
            ) <= model.tdu_e[e]

        # model.destination_duration_limit = pyomo.Constraint(model.E, rule=destination_duration_limit)

        def ad_hoc_charging_duration_rule(model, e):
            """Ensure ad-hoc charging duration matches unit commitment decision."""
            return sum(
                model.x_a[e, h] / self.inputs['r_a']
                for h in self.inputs['valid_hours'][e]
            ) <= model.tau_e[e]

        # model.ad_hoc_charging_duration_con = pyomo.Constraint(model.E, rule=ad_hoc_charging_duration_rule)

        # 4️⃣ Ad-Hoc Charging Linked to Commitment (`u_ab`)
        def enforce_ad_hoc_selection(model, e, h):
            """Ad-hoc charging (`x_a`) only when `u_ab[e]` is active."""
            return model.x_a[e, h] <= (model.u_ab[e]) * self.inputs['r_a']

        # model.enforce_ad_hoc_selection = pyomo.Constraint(model.EM, rule=enforce_ad_hoc_selection)

    def Solution(self):
        model_vars = self.model.component_map(ctype=pyomo.Var)
        serieses = []  # Collection to hold the converted series

        for k, v in model_vars.items():  # Loop over model variables
            values = v.extract_values()  # Extract solution values

            #  Skip empty variables
            if not values:
                print(f"WARNING: Variable '{k}' has no values. Skipping.")
                continue

            # Convert values to a Pandas Series
            s = pd.Series(values, index=values.keys())

            # If multi-indexed (i.e., tuple keys), reshape appropriately
            if isinstance(s.index[0], tuple):
                s = s.unstack(level=1)  # Reshape for readability
            else:
                s = pd.DataFrame(s)  # Convert Series to DataFrame for consistency

            # Assign proper column names
            s.columns = pd.MultiIndex.from_tuples([(k, t) for t in s.columns])
            serieses.append(s)

        # Combine all extracted values into a single DataFrame
        # self.solution = pd.concat(serieses, axis=1)
         # Combine all extracted values into a single DataFrame
        if serieses:  # ✅ Check if any non-empty variables are found
            self.solution = pd.concat(serieses, axis=1)
        else:
            print("WARNING: No variables had values. Solution is empty.")
            self.solution = pd.DataFrame()  # Return an empty DataFrame

    def Compute(self):

        """
        Computes:
        1. SOC evolution
        2. SIC (inconvenience metric)
        3. Charging details (cost, kWh, duration)
        """

        # Extract SOC values
        soc_values = []
        soc_after_trip_values = []

        for e in self.model.E:
            # Get SOC after charging
            soc_e = pyomo.value(self.model.ess_state[e]) if self.model.ess_state[e].value is not None else np.nan
            soc_values.append(soc_e / self.inputs['b_c'])

            # Get SOC after trip
            soc_trip = pyomo.value(self.model.soc_after_trip[e]) if self.model.soc_after_trip[e].value is not None else np.nan
            soc_after_trip_values.append(soc_trip / self.inputs['b_c'])

        # Prepend initial SOC
        soc = np.insert(soc_values, 0, self.inputs['s_i'] / self.inputs['b_c'])

        # Store SOC in solution
        self.solution["soc", 0] = soc[1:]  # SOC after charging
        self.solution["soc_start_trip", 0] = soc[:-1]  # SOC before the trip
        self.solution["soc_end_trip", 0] = soc_after_trip_values  # SOC after trip

        # Compute "inconvenience" metric (SIC)
        total_inconvenience = 0.0
        for e in self.model.E:
            # if pyomo.value(self.model.u_db[e]) > 0:
                penalty_e = self.inputs['c_db'][e] if self.inputs['is_other'][e] else 0.0
                for h in self.inputs['valid_hours'][e]:
                    if (e, h) not in self.model.EM:
                        continue
                    xd_val = pyomo.value(self.model.x_d[e, h])
                    total_inconvenience += penalty_e * xd_val

        self.sic = (total_inconvenience / 60) / (self.inputs['l_i'] / 1e3)

        # Compute charging details
        charging_costs = []
        charging_kwh_total = []
        charging_start_times = []
        charging_end_times = []
        hour_charging_details = []

        # Improved Charging Tracking with Tuple as Key
        for e in self.model.E:
            total_cost_e = 0.0
            hour_charging = defaultdict(float)  # Use defaultdict to track hourly charging

            for h in self.inputs['valid_hours'][e]:
                if (e, h) not in self.model.EM:
                    continue
                # Extract charging power at hour h
                xd_val = pyomo.value(self.model.x_d[e, h]) if self.model.x_d[e, h].value is not None else 0.0
                xa_val = pyomo.value(self.model.x_a[e, h]) if self.model.x_a[e, h].value is not None else 0.0
                event_energy_kw_h = (xd_val + xa_val) # * 2.7778e-7  # Stay in Joules, convert to kWh correctly

                if event_energy_kw_h > 1e-9:
                    # Convert hour_of_year to (day_of_week, hour_of_day)
                    day_of_week = (h // 24) % 7  # 0 = Monday, ..., 6 = Sunday
                    hour_of_day = h % 24
                    hour_charging[(day_of_week, hour_of_day)] += event_energy_kw_h

                # Get correct rate per hour
                if isinstance(self.inputs['residential_rate'], list):
                    res_rate = self.inputs['residential_rate'][min(h, 8759)]
                    com_rate = self.inputs['commercial_rate'][min(h, 8759)]
                else:
                    res_rate = self.inputs['residential_rate']
                    com_rate = self.inputs['commercial_rate']

                rate = res_rate if self.inputs['is_home'][e] else (
                    com_rate if self.inputs['is_work'][e] else self.inputs['other_rate'])

                total_cost_e += event_energy_kw_h * rate

            # Store charging details
            charging_kwh_total.append(sum(hour_charging.values()))
            charging_costs.append(total_cost_e)
            hour_charging_details.append(dict(hour_charging))  # Convert defaultdict to normal dict

            # Corrected Start and End Times using Tuple Keys
            if hour_charging:
                start_time = min(hour_charging.keys())  # (day_of_week, hour_of_day)
                end_time = max(hour_charging.keys())  # (day_of_week, hour_of_day)

                charging_start_times.append(start_time)
                charging_end_times.append(end_time)
            else:
                charging_start_times.append(np.nan)
                charging_end_times.append(np.nan)

        # Store final results
        self.solution["charging_kwh_total", 0] = pd.Series(charging_kwh_total).reindex(self.solution.index).tolist()
        self.solution["charging_cost", 0] = pd.Series(charging_costs).reindex(self.solution.index).tolist()
        self.solution["charging_start_time", 0] = pd.Series(charging_start_times).reindex(self.solution.index).tolist()
        self.solution["charging_end_time", 0] = pd.Series(charging_end_times).reindex(self.solution.index).tolist()
        self.solution["hour_charging_details", 0] = pd.Series(hour_charging_details).reindex(self.solution.index).tolist()