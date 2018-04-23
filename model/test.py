# m,n=list(map(int,raw_input().split()))
# List = []
# for i in range(0,m):
#     Input = list(raw_input().split())
#     if(Input[n]!="*"):
#         List.append(Input[0])
# if(len(List)==0):
#     print("None")
# else:
#     print(' '.join(List))

# import math
# r2= int(input())
# dis = 0
# record = 0
# r = int(math.sqrt(r2))
# List = []
# for x in range(0,r+1):
#     List.append(r2-x*x)
# for y in range(-r,r+1):
#     t = y*y
#     if t in List:
#         record = record+1
# record = record*2-2
# print(record)
# L,R=list(map(int,raw_input().split()))
# a = []
# for i in range(L,R+1):
#     items = str(i)
#     items_list = list(items)
#     items_set = set(items)
#     if len(items_set)==len(items_list):
#         a.append(i)
# print(len(a))

# def find(a,b,c,d,x):
#     arr = [a,b,c*2,d*2]
#     return pro1(arr,0,x)
#
# def pro1(a,index,mon):
#     res = 0
#     if mon==0:
#         return False
#     if index ==len(a)-1:
#         if (mon%a[index]==0):
#             return mon/a[index]
#         else:
#             return False
#     else:
#         res = []
#         for i in range(mon/a[index]+1):
#             temp = pro1(a,index+1,mon-a[index]*i)
#             if temp != False:
#                 res.append(temp+i)
#             else:
#                 continue
#     if len(res)==0:
#         return False
#     else:
#         return max(res)
#
# x = find(2,4,1,1,11)
# if x ==False:
#     x = 0
# print(x)
# def combinations(iterable, r):
#     # combinations('ABCD', 2) --> AB AC AD BC BD CD
#     # combinations(range(4), 3) --> 012 013 023 123
#     pool = tuple(iterable)
#     n = len(pool)
#     if r > n:
#         return
#     indices = list(range(r))
#     yield tuple(pool[i] for i in indices)
#     while True:
#         for i in reversed(range(r)):
#             if indices[i] != i + n - r:
#                 break
#         else:
#             return
#         indices[i] += 1
#         for j in range(i+1, r):
#             indices[j] = indices[j-1] + 1
#         yield tuple(pool[i] for i in indices)
#
# n = int(input())
# x = []
# y = []
# for i in range(n):
#     X,Y = list(map(int,input().split()))
#     x.append(X)
#     y.append(Y)
# time = combinations(range(n),3)
# count = 0
#
# for i in time:
#     print(i)
#     count = count+1
#     a1 = x[i[1]]-x[i[0]]
#     a2 = x[i[2]]-x[i[1]]
#     b1 = y[i[1]]-y[i[0]]
#     b2 = y[i[2]]-y[i[1]]
#     print(a1,a2,b1,b2)
#     if((a2==0)&(a1==0)):
#         count = count-1
#     if((b2==0)&(b1==0)):
#         count = count-1
#     if((a2!=0)&(b2!=0)):
#         if(a1/a2==b1/b2):
#             count = count-1
#     print(count)
# print(count)

# a = list(map(int,input().split()))
# b =a.copy()
# while(len(b)!=0):
#     print(len(b))
#     mina = min(b)
#     lenth = len(a)
#     for i in a:
#         j = i - mina
#         if(j==0):
#             b.remove(i)

# a = list(input())
# t = a.index(":")
# s = ""
# hour = int(s.join(a[0:t]))
# minute = int(s.join(a[t + 1:]))
# if minute == 60:
#     minute = 0
#
# minuteAngel = minute * 6
# if minute % 2 == 0:
#     hourAngel = (hour % 12) * 30 + int(minute * 0.5)
# else:
#     hourAngel = (hour % 12) * 30 + minute * 0.5
# angle = abs(hourAngel - minuteAngel)
# angle = min(360 - angle, angle)
# print(angle)
# globals res = []
# def partition(s):
#     res = findPalindrome(s, [])
#     return res
#
#
# def findPalindrome(s, plist):
#     for i in range(1, len(s) + 1):
#         if isPalindrome(s[:i]):
#             res = findPalindrome(s[i:], plist + [s[:i]])
#
#     if len(s) == 0:
#         res.append(plist)
#     return res
#
# def isPalindrome(s):
#     if s == s[::-1]:
#         return True
#     else:
#         return False
#
# s = input()
# a = partition(s)
# count = 0
# for i in range(len(a)):
#     for j in range(len(a[i])):
#         count+=1
# print(count)
#3
# def drump(end,num):
#     if (end==0):
#         return 0
#     i = 0
#     while(num[i]<end-i):
#         i +=1
#     return 1+drump(i,num)
#
# n = int(input())
# null = []
# for i in range(n):
#     temp = int(input())
#     null.append(temp)
#
# print (drump(n-1,null))
#
#
#
# #2
# a= int(input())
# b = int(input())
# c = a*b
# print(c)
#
#
# #1
# strlist = input()
# d= dict()
# for i in strlist:
#     if ord(i) in d.keys():
#         d[ord(i)]+=1
#     else:
#         d[ord(i)]=1
# count = len(strlist)
# result = []
# while(count>0):
#     for i in sorted(d.keys()):
#         if d[i]!=0:
#             result.append(chr(i))
#             d[i]-=1
#             count-=1
# s = ''
# print(s.join(result))

# def find(A,p):
#     high = len(A)-1
#     low = 0
#     if A[len(A)-1]<p:
#         return -1
#     else:
#         while(low!=high-1):
#             mid = (high+low)/2
#             if A[mid]<p:
#                 low = mid
#             elif A[mid]>=p:
#                 high = mid
#     return high
# A = [1,2,3,3,5,6,7]
# p = 3.5
# print(find(A,p))