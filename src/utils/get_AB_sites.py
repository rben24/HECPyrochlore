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

B_metals_s = []
B_metals = []
for el in elements:
    el = Element(el)
    isValid = False
    if el.is_metal:
        #if 4 in el.oxidation_states:
        #    isValid = True
        if 4 in el.common_oxidation_states:
            isValid = True
        if isValid:
            B_metals_s.append(el.symbol)
            B_metals.append(el)

print("---------------B Site Metals------------------")
for el in B_metals:
    print(f" el: {el.symbol} oxstate: {el.oxidation_states}")
print(B_metals_s)

A_metals_s = []
A_metals = []
for el in elements:
    el = Element(el)
    isValid = False
    if el.is_metal:
        #if 3 in el.oxidation_states:
        #    isValid = True
        if 3 in el.common_oxidation_states:
            isValid = True
        if isValid:
            A_metals_s.append(el.symbol)
            A_metals.append(el)

print("---------------A Site Metals------------------")
for el in A_metals:
    print(f" el: {el.symbol} oxstate: {el.oxidation_states}")
print(A_metals_s)
