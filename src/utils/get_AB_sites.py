import json
from pymatgen.core import Element

with open("periodic_table.json", "r") as f:
    ptjson = json.load(f)
elements = []
for key in ptjson:
    if key.upper() == "_UNIT":
        pass
    elif key.upper() == "D" or key.upper() == "T":
        pass
    else:
        elements.append(key)

B_metals_s_4 = []
B_metals_s_5 = []
B_metals = []
for el in elements:
    el = Element(el)
    isValid = False
    if el.is_metal:
        #if 4 in el.oxidation_states:
        #    isValid = True
        if 4 in el.common_oxidation_states:
            isValid = True
            B_metals_s_4.append(el.symbol)
        if 5 in el.common_oxidation_states:
            isValid = True
            B_metals_s_5.append(el.symbol)
        if isValid:
            B_metals.append(el)

print("---------------B Site Metals------------------")
for el in B_metals:
    print(f" el: {el.symbol} oxstate: {el.oxidation_states}")
print(f"+4: {B_metals_s_4}")
print(f"+5: {B_metals_s_5}")

A_metals_s_3 = []
A_metals_s_2 = []
A_metals = []
for el in elements:
    el = Element(el)
    isValid = False
    if el.is_metal:
        #if 3 in el.oxidation_states:
        #    isValid = True
        if 3 in el.common_oxidation_states:
            isValid = True
            A_metals_s_3.append(el.symbol)
        if 2 in el.common_oxidation_states:
            isValid = True
            A_metals_s_2.append(el.symbol)
        if isValid:
            A_metals.append(el)

print("---------------A Site Metals------------------")
for el in A_metals:
    print(f" el: {el.symbol} oxstate: {el.oxidation_states}")
print(f"+3: {A_metals_s_3}")
print(f"+2: {A_metals_s_2}")
