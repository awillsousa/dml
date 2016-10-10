from security import Stock, Holding
from invest import Investor
from market import Market, trade, create_market_securities, securities_tot_value

sec_1 = Stock('A', 10.0)
sec_2 = Stock('B', 11.0)

holding_1 = Holding(sec_1, 4)
holding_2 = Holding(sec_2, 3)

print("Holding value = " + str(holding_1.value()))

investor_1 = Investor("Inv1", cash = 13., holdings = {holding_1.security.id : holding_1 , holding_2.security.id : holding_2})

print("Investor's securities value = " + str(investor_1.securities_value()))
print("Investor's total assets value = " + str(investor_1.tot_assets_value()))

sec_3 = Stock('C', 8.0)
sec_4 = Stock('D', 11.0)

holding_3 = Holding(sec_3, 1)
holding_4 = Holding(sec_4, 4)

investor_2 = Investor("Inv2", cash = 4., holdings = {holding_3.security.id : holding_3 , holding_4.security.id : holding_4})

market = Market(investors = [investor_1 , investor_2])

print("Market total cash = " + str(market.tot_cash()))
print("Market total stock value = " + str(market.tot_stock_value()))

print("Buying one stock from holding")
print("Holding value before buy= " + str(holding_1.value()))
print(holding_1.buy(1))
print("Holding value after buy = " + str(holding_1.value()))
print("Selling one stock from holding")
print(holding_1.sell(1))
print("Holding value after sell= " + str(holding_1.value()))

print("Investor buying one stoock")
print("Investor's securities value before buy = " + str(investor_1.securities_value()))
print("Investor's cash before buy = " + str(investor_1.cash))
print("Investor's total assets value before buy= " + str(investor_1.tot_assets_value()))
holding_5 = Holding(sec_1, 1)
investor_1.buy_sec(holding_5)
print("Investor's securities value after buy = " + str(investor_1.securities_value()))
print("Investor's cash after buy = " + str(investor_1.cash))
print("Investor's total assets value after buy= " + str(investor_1.tot_assets_value()))

print("Investor selling one stoock")
print("Investor's securities value before sell = " + str(investor_1.securities_value()))
print("Investor's cash before sell = " + str(investor_1.cash))
print("Investor's total assets value before sell= " + str(investor_1.tot_assets_value()))
holding_6 = Holding(sec_1, 1)
investor_1.sell_sec(holding_6)
print("Investor's securities value after sell = " + str(investor_1.securities_value()))
print("Investor's cash after sell = " + str(investor_1.cash))
print("Investor's total assets value after sell= " + str(investor_1.tot_assets_value()))

print("Market total cash = " + str(market.tot_cash()))
print("Market total stock value = " + str(market.tot_stock_value()))

investor_1 = Investor("Inv1", cash = 53., holdings =
                    {holding_1.security.id : holding_1
                        , holding_2.security.id : holding_2
                        , holding_3.security.id : holding_3
                        , holding_4.security.id: holding_4})

holding_7 = Holding(sec_1, 5)
holding_8 = Holding(sec_2, 2)
holding_9 = Holding(sec_3, 1)
holding_10 = Holding(sec_4, 4)

investor_2 = Investor("Inv2", cash=44., holdings=
                    {holding_7.security.id: holding_7
                        , holding_8.security.id: holding_8
                        , holding_9.security.id: holding_9
                        , holding_10.security.id: holding_10})

print("Trading between investors")
print("Seller's cash before trade =" + str(investor_2.cash))
print("Seller's securities value before trade=" + str(investor_2.securities_value()))
print("Buyer's's cash before trade =" + str(investor_1.cash))
print("Buyers's securities value before trade=" + str(investor_1.securities_value()))
trade(investor_2 , investor_1, Holding(sec_1, 2))
print("Seller's cash after trade =" + str(investor_2.cash))
print("Seller's securities value after trade=" + str(investor_2.securities_value()))
print("Buyer's's cash after trade =" + str(investor_1.cash))
print("Buyers's securities value after trade=" + str(investor_1.securities_value()))

quots = create_market_securities()
print(len(quots))
tot_secs_val = securities_tot_value(quots)
print("Market total securities value = " + str(tot_secs_val))




