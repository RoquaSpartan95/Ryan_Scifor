def reverse_string(str):  
    str1 = ""   # Declaring empty string to store the reversed string  
    for i in str:  
        str1 = i + str1  
    return str1      
     
str = input("enter your string:")    
print("The original string is: ",str)  
print("The reverse string is",reverse_string(str)) 