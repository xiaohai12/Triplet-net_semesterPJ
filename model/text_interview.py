# def dicesSum(self, n):
#     # Write your code here
#     if n == 0: return None
#     result = [
#         [1, 1, 1, 1, 1, 1],
#     ]
#     # if n == 1: return result[0]
#
#     for i in range(1, n):
#         x = 5 * (i + 1) + 1
#         result.append([0 for _ in range(x)])
#
#         for j in range(x):
#             if j < 6:
#                 result[i][j] = (sum(result[i - 1][0:j + 1]))
#             elif 6 <= j <= 3 * i + 2:
#                 result[i][j] = (sum(result[i - 1][j - 5:j + 1]))
#             else:
#                 break
#         left = 0
#         right = len(result[i]) - 1
#         while left <= right:
#             result[i][right] = result[i][left]
#             left += 1
#             right -= 1
#
#     res = result[-1]
#     all = float(sum(res))
#     other = []
#
#     for i, item in enumerate(res):
#         pro = round(item/all)
#
#         #pro = item / all
#         other.append([n + i, pro])
#     return other
#
# def round(self, num):
#
#     num = num * 100
#     num = int(2 * num) / 2 + int(2 * num) % 2
#     num = num / 100.0
#     return num
#
#
# a = dicesSum(1)

作者：浪飘
链接：https://www.nowcoder.com/discuss/75796?type=2&order=0&pos=14&page=1
来源：牛客网

import java.util.Scanner; 
    public class 华为2 {   static int[] dayl = { 12, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30 };  
    public static void main(String[] args){   Scanner sc = new Scanner(System.in );  
    int a = sc.nextInt();  
    int b = sc.nextInt();  
    int c = Result(a, b);  
    if(c>0)   System.out.println(c);  
    else    System.out.println(-1);   }  
    private static int Result(int year, int weeks){
       int count = 0;   int days = 0;  
        try{   if(weeks <=6 && weeks>=0 && year>=0 && year<=400){ 
  for(int i=1900; i<1900+year; i++){ 
  days += i==1900?0:(runnian(i-1)?366:365);  
int day = days;  
    for(int j=1; j<=12; j++){ 
    days += getDay(i,j);  
    if((days-(weeks-1))%7==0){   count++;   }   } 
      days = day;  
    }  
    if(weeks<0 || weeks>6 || year<0 || year>400)   return -1;   }}catch (Exception e)
{   return -1;   }   return count;   }  
private static int getDay(int i, int j)
    {   if(!runnian(i)){   return dayl[j-1];   } 
    return  j == 3? 29 : dayl[j-1];   }  
private static boolean runnian(int i)
{   return (i%4 == 0 && i % 100 !=0)||i%400 == 0;   }  }