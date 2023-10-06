#here is all the logic behind strategic decision making
import alive_progress
import time

class strategy:
    def __init__(self, squad, *args, **kwargs):
        self.squad = squad
        #self.y = y
        #self.t = t
        #squad - list of models of class <predictor>


    def hmbsa(self, bit_of_data, price): #how_much_to_buy_or_sell_all
        #bit_of_data - carefully engineered(corresponding with the input_size of each model in the squad) bit of data that fits for the squad to process
        #price - price of the seccurity RIGHT NOW
        ps = [m.make_prediction(bit_of_data) for m in self.squad] 
        n = 1
        shares = 0
        if ps[0][1] > price*1.0005:
            for p in range(1, len(ps)):
                if ps[p][1] > ps[p-1][1]:
                    n = p+1
        
        if n > 1: #buy in advance
            shares += n
            
        if ps[0][0] > price*1.0005: # buy for the next day
            shares += 2
            
        elif ps[0][1] > price*1.0005 and ps[0][0] <= price*1.0005: #buy if the next step everything is not going to be so bright
            shares += 1
            
        if ps[0][1] < price*1.0005: #sell all if the next day the price is expected to fall the next step
            return 0 #shares to sell

        return shares #shares to buy

class testing_agent:
    def __init__(self, strat, data, budget=0, shares=0):
        self.strat = strat #strategy_class.strategy() obj
        self.data = data #tuple of training data bits as after predictor_class.get_trainingdata()
        self.budget = budget #just a number, int or float
        self.squad = self.strat.squad #list of predictor_class.predictor() objs
        self.shares = shares
        self.pnls = []

    def test_strat(self):
        with alive_progress.alive_bar(len(self.data[0])-1, force_tty=True, spinner='stars') as bar:
            for i in range(len(self.data[1])):
                time.sleep(0.0005)
                data_bit = self.data[0][i:i+1]
                price = self.data[1][i-1]
                # right now is the i moment, in the labels_s` realm it`s i-1st moment
                shares_now = self.strat.hmbsa(data_bit, price)
                if shares_now == 0:
                    self.budget += self.shares*price
                    self.shares = 0
                else:
                    self.shares += shares_now
                    self.budget -= shares_now*price
                self.pnls.append(self.budget)
                bar()