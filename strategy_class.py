#here is all the logic behind strategic decision making

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



