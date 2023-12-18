myList = [1, 2, 3, 4, 5, 6, 7, 8, 9]
print("The given list is:")
print(myList)
list_len=len(myList)
sum=0
for i in range(list_len):
    sum=sum+myList[i]

print("Sum of all the elements in the list is:", sum)