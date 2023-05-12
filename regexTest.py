import re

s = \
    "Read the abstract section in the article. Answer the following questions. Enumerate your responses:" \
    "1. First, print out the abstract" \
    "2. Based on the abstract, fill in the blank what is the main dependent variable in the text? Be specific and concise." \
    "3. Based on the abstract, fill in the blank what is the main independent variable in the text? Be specific and concise." \
    "4. Based on the abstract, fill in the blank what is the sample in the text? Be specific and concise." \
    "5. Based on the abstract, fill in the blank what is the population to which these findings apply to? Be specific and concise." \
    "6. Based on the abstract, fill in the blank what is the experimental manipulation? Be specific and concise." \
    "7. Based on the abstract, fill in the blank what is mechanism of the experimental manipulation? Be specific and concise." \
    "8. Based on the abstract, fill in the blank what is the temporal context in which these findings apply? Be specific and concise." \
    "9. Based on the abstract, fill in the blank what is the main reccommendation from this text? Be specific and concise."


def getregex(s):
    return s + ". ([^" + str(int(s) + 1) + "]+)"

full_ab_arr = []

m = re.search(getregex("7"), s)
full_ab_arr.append(m.group(1)) if m is not None else None


print(full_ab_arr)