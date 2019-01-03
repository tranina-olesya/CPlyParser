from cTreeParser import *
import os
import sys


s1 = '''
        int plus(int a, int b)
        {
            int res = a + b;
            return res;
        }
        double minus(double a)
        {
            return a - 1;
        }
        
        double solve(int a)
        {
            if(a > 4)
                return 0;
            else
                return minus(plus(a, 9));
        }
        
        void main() { 
            output_double(solve(input_int()));
        }
    '''


s8= '''
    //int a, b = 0;
    void main()
    {
        string a = "hello";
        for (int i = 0; i<sizeof(a); i++)
        {
            output_char(a[i]);
        }
    }   
'''

test = '''
    void main()
    {
        int a = input_int(), b = input_int();
        bool ab = !a && a>b || b-a;
        if (ab)
            output_int(1);
        if ( !(a==b) && a>b && 1 &&a)
           output_int(2);
        else
           output_int(3);
        if (a>b || !a || 0 || b)
           output_int(4);
	}
    
'''

test2 = '''
/*int factorial(int n)
    {
        int s = 1;
        for (int i = 2; i <= n; i++)
            s *= i;
        return s;
    }
    */
    int f(int n) 
    {
      if (n == 1 || n == 2)
        {return 1;
        int a =0;
        output_int(a);
         }
      return f(n - 1) + f(n - 2);
    }
    int main()
    {
      int n = input_int();
      for (int i = 1; i <= n; i++)
        output_int(f(i));
      return 0;
    }
    /*
    void sort(){
        int[] arr = new int[2];
        int size = 5;
        arr = new int[5];
        
        for (int i = 0; i < size; i++) 
            arr[i]=input_int();
            
        for (int i = 0; i < size - 1; i++) {
            for (int j = 0; j < size - i - 1; j++) {
                if (arr[j] > arr[j + 1]) {
                    int temp = arr[j];
                    arr[j] = arr[j + 1];
                    arr[j + 1] = temp;
                }
            }
        }
        for (int i = 0; i < 5; i++) 
            output_int(arr[i]);
    }*/

'''
test31 = '''
    int f(int n) 
    {
      if (n == 1 || n == 2)
        {return 1;
        int a =0;
        output_int(a);
         }
      return f(n - 1) + f(n - 2);
    }
    int main()
    {
      int n = input_int();
      for (int i = 1; i <= n; i++)
        output_int(f(i));
      return 0;
    }
'''

test3 = '''
    int factorial(int n)
    {
        int s = 1;
        for (int i = 2; i <= n; i++)
            s *= i;
        return s;
    }
    int main()
    {
      int n = input_int();
      for (int i = 1; i <= n; i++)
        output_int(factorial(i));
      return 0;
    }
'''


test4 = '''
    void main()
    {
        
	char a = '\n';
	to_str(a);
    }
'''
test5 = '''
    int[] b = new int[] {3,1,42,2};
    int a=0, e = 2;
    void main(){
        int[] a = new int[] {1,4,2};
        a = b;
        for(int i = 0; i<4; i++)
            output_int(a[i]);
    }
'''


#build_tree(test4)
code_generate(sys.argv[1])
