# STRING MANIPULATIONS

#1.	Create a string “Grow Gratitude”.
my_string = 'Grow Gratitude'
print(my_string)

#access the letter “G” of “Growth”
my_string1 = 'Growth'
print(my_string1)
print(my_string1[0])

#find the length of the string
print(len(my_string1))

#Count how many times “G” is in the string
print(my_string1.count('G'))


#2.	Create a string 
my_string2 = 'Being aware of a single shortcoming within yourself is far more useful than being aware of a thousand in someone else.'

#a)	Count the number of characters in the string.
print(len(my_string2))

#3.	Create a string 
my_string3 = "Idealistic as it may sound, altruism should be the driving force in business, not just competition and a desire for wealth"

#a)	get one char of the word
print(my_string3[0])

#b)	get the first three char
print(my_string3[0:3])

#c)	get the last three char
print(my_string3[-3:])

#create a string "stay positive and optimistic".
my_string4 = "stay positive and optimistic"

#code to split on whitespace.
my_string4.split()

#code to find if the string starts with “H”

print(my_string4.startswith('H'))

#code to find if The string ends with “d”
print(my_string4.endswith('d'))

#code to find if The string ends with “c”
print(my_string4.endswith('c'))

#code to print " 🪐 " one hundred and eight times. (only in python)
print("🪐"*108)

#7.	Create a string “Grow Gratitude” and write a code to replace “Grow” with “Growth of”
my_string5 = 'Grow Gratitude'
print(my_string5.replace("Grow", "Growth"))

#reversed order string
my_string6 = '.elgnujehtotniffo deps mehtfohtoB .eerfnoilehttesotseporeht no dewangdnanar eh ,ylkciuQ .elbuortninoilehtdecitondnatsapdeklawesuomeht ,nooS .repmihwotdetratsdnatuotegotgnilggurts saw noilehT .eert a tsniagapumihdeityehT .mehthtiwnoilehtkootdnatserofehtotniemacsretnuhwef a ,yad enO .ogmihteldnaecnedifnocs’esuomeht ta dehgualnoilehT ”.emevasuoy fi yademosuoyotplehtaergfo eb lliw I ,uoyesimorp I“ .eerfmihtesotnoilehtdetseuqeryletarepsedesuomehtnehwesuomehttaeottuoba saw eH .yrgnaetiuqpuekow eh dna ,peels s’noilehtdebrutsidsihT .nufroftsujydobsihnwoddnapugninnurdetratsesuom a nehwelgnujehtnignipeelsecno saw noil A'
print(my_string6[::-1])
