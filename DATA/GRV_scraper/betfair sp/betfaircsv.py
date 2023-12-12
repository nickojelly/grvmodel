import pickle


betfairCsvLinks = [
"https://promo.betfair.com/betfairsp/prices/dwbfgreyhoundwin01102022.csv",
"https://promo.betfair.com/betfairsp/prices/dwbfgreyhoundwin01092022.csv",
"https://promo.betfair.com/betfairsp/prices/dwbfgreyhoundwin01082022.csv",
"https://promo.betfair.com/betfairsp/prices/dwbfgreyhoundwin01072022.csv",
"https://promo.betfair.com/betfairsp/prices/dwbfgreyhoundwin01062022.csv",
"https://promo.betfair.com/betfairsp/prices/dwbfgreyhoundwin01052022.csv",
"https://promo.betfair.com/betfairsp/prices/dwbfgreyhoundwin01042022.csv",
"https://promo.betfair.com/betfairsp/prices/dwbfgreyhoundwin01032022.csv",
"https://promo.betfair.com/betfairsp/prices/dwbfgreyhoundwin01022022.csv",
"https://promo.betfair.com/betfairsp/prices/dwbfgreyhoundwin01012023.csv",
"https://promo.betfair.com/betfairsp/prices/dwbfgreyhoundwin01012022.csv",
"https://promo.betfair.com/betfairsp/prices/dwbfgreyhoundwin30012023.csv",
"https://promo.betfair.com/betfairsp/prices/dwbfgreyhoundwin29012023.csv",
"https://promo.betfair.com/betfairsp/prices/dwbfgreyhoundwin28012023.csv",
"https://promo.betfair.com/betfairsp/prices/dwbfgreyhoundwin27012023.csv",
"https://promo.betfair.com/betfairsp/prices/dwbfgreyhoundwin26012023.csv",
"https://promo.betfair.com/betfairsp/prices/dwbfgreyhoundwin25012023.csv",
"https://promo.betfair.com/betfairsp/prices/dwbfgreyhoundwin24012023.csv",
"https://promo.betfair.com/betfairsp/prices/dwbfgreyhoundwin23012023.csv",
"https://promo.betfair.com/betfairsp/prices/dwbfgreyhoundwin22012023.csv",
"https://promo.betfair.com/betfairsp/prices/dwbfgreyhoundwin21012023.csv",
"https://promo.betfair.com/betfairsp/prices/dwbfgreyhoundwin20012023.csv",
"https://promo.betfair.com/betfairsp/prices/dwbfgreyhoundwin19012023.csv",
"https://promo.betfair.com/betfairsp/prices/dwbfgreyhoundwin18012023.csv",
"https://promo.betfair.com/betfairsp/prices/dwbfgreyhoundwin17012023.csv",
"https://promo.betfair.com/betfairsp/prices/dwbfgreyhoundwin16012023.csv",
"https://promo.betfair.com/betfairsp/prices/dwbfgreyhoundwin15012023.csv",
"https://promo.betfair.com/betfairsp/prices/dwbfgreyhoundwin14012023.csv",
"https://promo.betfair.com/betfairsp/prices/dwbfgreyhoundwin13012023.csv",
"https://promo.betfair.com/betfairsp/prices/dwbfgreyhoundwin12012023.csv",
"https://promo.betfair.com/betfairsp/prices/dwbfgreyhoundwin11012023.csv",
"https://promo.betfair.com/betfairsp/prices/dwbfgreyhoundwin10012023.csv",
"https://promo.betfair.com/betfairsp/prices/dwbfgreyhoundwin09012023.csv",
"https://promo.betfair.com/betfairsp/prices/dwbfgreyhoundwin08012023.csv",
"https://promo.betfair.com/betfairsp/prices/dwbfgreyhoundwin07012023.csv",

]


with open("betfairCsvLinks2.npy", "wb") as fp:   #Pickling
    
    pickle.dump(betfairCsvLinks, fp)

print("done")