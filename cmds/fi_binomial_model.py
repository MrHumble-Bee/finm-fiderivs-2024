import pandas as pd
import numpy as np

import pandas as pd
import numpy as np


class FIBinomialModel():
    def __init__(self, face_value: float=100) -> None:
        self.__face_value: float = face_value
        self.__rate_tree: pd.DataFrame = None
        self.__bond_tree: pd.DataFrame = None
        self.__swap_tree: pd.DataFrame = None

        self.__dt: float = None
        self.__T: float = None
    
    @property
    def rate_tree(self) -> pd.DataFrame:
        return self.__rate_tree

    @rate_tree.setter
    def rate_tree(self, new_rate_tree: pd.DataFrame) -> None:
        if isinstance(new_rate_tree, pd.DataFrame):
            self.__rate_tree = new_rate_tree
        else:
            raise TypeError("Can only set rate_tree attribute to pd.DataFrame type only.")
        
    @property
    def bond_tree(self) -> pd.DataFrame:
        return self.__bond_tree
    
    @bond_tree.setter
    def bond_tree(self, new_bond_tree: pd.DataFrame) -> None:
        if isinstance(new_bond_tree, pd.DataFrame):
            self.__bond_tree = new_bond_tree
        else:
            raise TypeError("Can only set bond_tree attribute to pd.DataFrame type only.")

    @property
    def p_star(self):
        if (self.__rate_tree is None or self.__bond_tree is None or self.__dt is None or self.__T is None):
            raise Exception("Compute rate tree and bond tree first.")
        
        A = np.exp(self.__rate_tree.iloc[0,0] * self.__dt)

        pstar = (A * self.__bond_tree.loc[0,0] - self.__bond_tree.loc[1,self.__dt])/(self.__bond_tree.loc[0,self.__dt] - self.__bond_tree.loc[1,self.__dt])
        return pstar

    def initialize_empty_rate_tree(self, dt, T) -> None:
        self.__dt = dt
        self.__T = T
        timegrid = pd.Series((np.arange(0,round(T/dt)+1)*dt).round(6),name='time',index=pd.Index(range(round(T/dt)+1),name='state'))
        tree = pd.DataFrame(dtype=float,columns=timegrid,index=timegrid.index)
        self.__rate_tree = tree

    def compute_bond_tree(self, t0_price: float=None) -> None:
        maturity = (self.__rate_tree.columns[-1] - self.__rate_tree.columns[-2]) + self.__rate_tree.columns[-1]
        bond_tree_cols = list(self.__rate_tree.columns.copy().values)
        bond_tree_cols.append(maturity)
        print(bond_tree_cols)
        bond_tree = pd.DataFrame(dtype=float, index=pd.RangeIndex(0, len(bond_tree_cols)), columns=bond_tree_cols)
        bond_tree.index.name = self.__rate_tree.index.name
        bond_tree.columns.name = self.__rate_tree.columns.name
        
        forward_time: float = bond_tree.columns[-1]
        for time in reversed(bond_tree.columns):
            if time == maturity:
                bond_tree[time] = float(self.__face_value)
                forward_time = time
                continue
            bond_tree[time] = np.exp(-self.__rate_tree[time]*(forward_time-time)) * ((bond_tree[forward_time] + bond_tree[forward_time].shift(-1)) / 2)
            forward_time = time
        if t0_price:
            bond_tree.iloc[0,0] = t0_price
        self.__bond_tree = bond_tree

    def compute_swap_tree(self, payoff_func) -> pd.DataFrame:
        Z = np.exp(-self.__rate_tree.iloc[0,0] * self.__dt)
        swap_tree = pd.DataFrame(index=self.__rate_tree.index, columns=self.__rate_tree.columns, dtype=float)
        swap_tree[self.__dt] = payoff_func(self.__rate_tree[self.__dt])
        swap_tree.loc[0,0] = Z * np.array([self.p_star,1-self.p_star])@ swap_tree[self.__dt].values
        self.__swap_tree = swap_tree
        return swap_tree


    def display_rate_tree(self) -> pd.DataFrame:
        return self.__rate_tree.style.format('{:.4%}',na_rep='').format_index('{:.2f}',axis=1)
    
    def display_bond_tree(self) -> pd.DataFrame:
        return self.__bond_tree.style.format('$ {:.4}',na_rep='').format_index('{:.2f}',axis=1)