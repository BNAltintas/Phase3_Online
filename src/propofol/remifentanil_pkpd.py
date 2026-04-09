import numpy as np
from scipy import integrate
from scipy.optimize import brentq

from propofol.protocols import Patient

class EleveldPatient():
    def __init__(self, age: float, weight: float, height: float, sex: str):
        """
        Initialize patient parameters for PK/PD modelling.

        Parameters
        ----------
        age : float
            Age in years.
        weight : float
            Body weight in kilograms.
        height : float
            Height in centimeters.
        sex : str
            Biological sex ('male' or 'female', case-insensitive).
        """
        # Reference person
        self.age_ref = 35  # years
        self.sex_ref = 'male'
        self.weight_ref = 70  # kg
        self.height_ref = 170  # cm
        self.bmi_ref = self.weight_ref / (self.height_ref / 100.)**2

        # individual setting
        self.age = age  # years
        self.sex = sex
        self.weight = weight
        self.height = height
        self.bmi = bmi if (bmi is not None) else self.weight / (self.height / 100.)**2 
    
    
class EleveldPK(EleveldPatient):
    def __init__(self, patient: Patient, use_bsv: bool = False):
        self.patient = patient
        self.patient_ref = EleveldPatient(age=35, weight=70, height = 170, sex ='male')
        self.use_bsv = use_bsv
        self._set_params()

    def _set_params(self):
        self.Theta1 = 5.81 # L (V1)
        self.Theta2 = 8.82 # L (V2)
        self.Theta3 = 5.03 # L (V3)
        self.Theta4 = 2.58 # L/min (CL)
        self.Theta5 = 1.72 # L/min (Q2)
        self.Theta6 = 0.124 # L/min (Q3)
        self.Theta7 = 2.88 # (Θ1: defines the weight at which 50% of maturation of CL is complete)
        self.Theta8 = -0.00554  # (Θ2 defines the change in V1, Q2, and Q3 with age)
        self.Theta9 = -0.00327  # (Θ3 defines the change in V2 and CL with age)
        self.Theta10 = -0.0315  # (Θ4 defines the change in V3 with age)
        self.Theta11 = 0.470    # (Θ5 defines the change in V2, CL, and Q2 for females aged 12 to 45 yr)
        self.Theta12 = -0.0260    # (Θ6 defines the deviation of V3 from theoretical allometric scaling)

        # Set eta
        self.eta1_var = 0.104
        self.eta2_var = 0.115
        self.eta3_var = 0.810
        self.eta4_var = 0.0197
        self.eta5_var = 0.0547
        self.eta6_var = 0.285

        self.eta1 = 0
        self.eta2 = 0
        self.eta3 = 0
        self.eta4 = 0
        self.eta5 = 0
        self.eta6 = 0
        self.epsilon = 0

        # Pharmacokinetic parameter equations
        self.V1 = self.Theta1 * self.SIZE() * self.f_ageing(self.Theta8) * np.exp(self.eta1)  # L
        self.V2 = self.Theta2 * self.SIZE() * self.f_ageing(self.Theta9) * self.KSEX() * np.exp(self.eta2)  # L
        self.V3 = self.Theta3 * self.SIZE() * self.f_ageing(self.Theta10) * np.exp(self.Theta12 * (self.weight - self.weight_ref)) * np.exp(self.eta3)  # L
        self.CL = self.Theta4 * self.SIZE()**0.75 * (self.KMAT(self.weight)/self.KMAT(self.weight_ref)) * self.KSEX() * self.f_ageing(self.Theta9) * np.exp(self.eta4)  # L/min
        self.Q2 = self.Theta5 * (self.V2 / self.Theta2)**0.75 * self.f_ageing(self.Theta8) * self.KSEX() * np.exp(self.eta5)  # L/min
        self.Q3 = self.Theta6 * (self.V3 / self.Theta3)**0.75 * self.f_ageing(self.Theta8) * np.exp(self.eta6)  # L/min

        # Elimination rate constants
        self.k10 = self.CL / self.V1  # 1/min
        self.k12 = self.Q2 / self.V1  # 1/min
        self.k13 = self.Q3 / self.V1  # 1/min
        self.k21 = self.Q2 / self.V2  # 1/min
        self.k31 = self.Q3 / self.V3  # 1/min

    def SIZE(self):
        return (self.f_al_sallami(self.age, self.weight, self.bmi) / self.f_al_sallami(self.age_ref, self.weight_ref, self.bmi_ref))

    def f_ageing(self, x):
        return np.exp(x * (self.age - self.age_ref))
    
    def f_sigmoid(self, x, E50, lmbda):
        return x**lmbda / (x**lmbda + E50**lmbda)
    
    def KMAT(self, weight):
        return self.f_sigmoid(weight, self.Theta7, 2)
    
    def f_al_sallami(self, age, weight, bmi):
        # Fat free mass (FFM) in kg 
        if self.sex == 'male':
            return (0.88 + (1 - 0.88) / (1 + (age / 13.4)**(-12.7))) * (9720 * weight) / (6680 + 216 * bmi)
        elif self.sex == 'female':
            return (1.11 + (1 - 1.11) / (1 + (age / 7.1)**(-1.1))) * (9720 * weight) / (8780 + 244 * bmi)
    
    def KSEX(self):
        if self.sex == 'male':
            return 1
        elif self.sex == 'female':
            return 1 + self.Theta11 * self.f_sigmoid(self.age, 12, 6) * (1 - self.f_sigmoid(self.age, 45, 6))
        
    def draw_eta(self):
        self.eta1 = np.random.normal(loc=0, scale=np.sqrt(self.eta1_var))
        self.eta2 = np.random.normal(loc=0, scale=np.sqrt(self.eta2_var))
        self.eta3 = np.random.normal(loc=0, scale=np.sqrt(self.eta3_var))
        self.eta4 = np.random.normal(loc=0, scale=np.sqrt(self.eta4_var))
        self.eta5 = np.random.normal(loc=0, scale=np.sqrt(self.eta5_var))
        self.eta6 = np.random.normal(loc=0, scale=np.sqrt(self.eta6_var))
    
    def draw_epsilon(self):
        self.epsilon = np.random.normal(loc=0, scale=1)

    def Cobs(self, x):
        # Observation model
        return x + np.exp(self.epsilon)

class PKPD(EleveldPatient):
    def __init__(self, age, weight, height, sex: str, bmi):
        # Patient characteristics
        super().__init__(age, weight, height, sex = sex, bmi=bmi)

        # pk model
        self.pk = EleveldPK(age, weight, height, sex, bmi)
        #self.pd = EleveldPD(age, weight, height, sex, bmi)


    def derivative(self, X, t, k10, k12, k13, k21, k31):
        A1, A2, A3 = X

        dotA1 = (- (k10 + k12 + k13) * A1 
                 + k21 * A2 + k31 * A3
        )
        dotA2 = k12 * A1 - k21 * A2
        dotA3 = k13 * A1 - k31 * A3
        #dotCe = ke0 * (A1 / V1 - Ce)  # Since Ae does not affect Ax this could be moved out.
        return np.array([dotA1, dotA2, dotA3])
    
    def solve_ode(self, t = np.linspace(0, 15, 15*60+1), X0  = [1, 0, 0]):
        res = integrate.odeint(self.derivative, X0, t,
                        args = (self.pk.k10, 
                                self.pk.k12, 
                                self.pk.k13, 
                                self.pk.k21, 
                                self.pk.k31,) 
                                #self.pd.ke0)
        )
        return res.T  # A1[mg], A2[mg], A3[mg] with units [mg, mg, mg]
    
    def __call__(self, t = np.linspace(0, 15, 15*60+1), X0  = [1, 0, 0]):
        return self.solve_ode(t, X0)
