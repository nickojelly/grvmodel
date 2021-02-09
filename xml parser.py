import os
import xml.etree.ElementTree as ET
import pandas as pd

#working results 13 may
#fix split times to be correct, 1st time in array is not always smallest
#

if __name__ == '__main__':



    root = ET.parse("sample_results.xml").getroot()

    print(root.attrib)


    track = root[0]
    print("track =", track.text)

    date = root[1]
    print("date =", date.text)

    for race in root.iter('Race'):
        print("race number = ",race[0].text)
        print("Distance =", race[3].text, "\n")
        print("place name box split split_margin runtime")

        places = []
        ids = []
        boxes = []
        splittimes = []
        split_margin =[]
        runtimes = []

        for dog in race.iter('Dog'):
            if not dog.attrib["id"]:
                continue
            if (dog[0].text == "R")or(dog[0].text == "S"):
                #print("scratched check working")
                continue
            places.append(dog[0].text)
            ids.append(dog.attrib["id"])
            boxes.append(dog[3].text)
            splittimes.append(dog[13].text)
            runtimes.append(dog[14].text)

            #print(place.text, id, box.text, split.text, runtime.text)
        for i in range(len(splittimes)):
            if not splittimes[0]:
                split_margin.append("N/A")
                continue
            split_margin.append(float(splittimes[i])-float(splittimes[0]))

        for i in range(len(places)):
            try:
                print(places[i], ids[i], boxes[i], splittimes[i], "%.2f" % split_margin[i], runtimes[i])
            except :
                print(places[i], ids[i], boxes[i], splittimes[i], split_margin[i], runtimes[i])

        print("\n------------------------\n")
        zipall = list(zip(places,ids,boxes,splittimes,split_margin,runtimes))
        print(zipall)
        df = pd.DataFrame(zipall)
        print(df)

    #for meet in root:
    # print(meet.tag, meet.attrib, "\n")
    #     #break
    #     for race in meet:
    #         print(race.tag, race.attrib)
    #
    #         print("\n------------------\n")
    #         for dog in race:
    #             print( dog.tag, dog.attrib, "\n")


        #break
