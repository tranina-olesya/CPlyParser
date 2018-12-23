from cTreeParser import *
import os


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
            if(minus(a) > 4)
                return 0;
            else
                return minus(plus(a, 9));
        }
        
        void main() { 
            output_double(solve(input_int()));
        }
    '''

s2 = '''
        int a, b;
        void test1(int w) 
        {
            int g, g2 = g;
    
            /* comment 1
            c = input();
            */
            for (int i = 0, j = 8; ((i <= 5)) && g; i++)
                for(; a < b;)
                    if (a > 7 + b) {
                        int c = a + b * (2 - 1) + 0;  // comment 2
                        string ab = "98\tура";
                    }
            for(;;);
        }
    '''

s3 = '''
        void test2() 
        {
            int a;
            
            {
                int a;
            }   
            while (a==0){
                if (a > 23 + 1 && 2) 
                    int a = 0;
                else 
                    if (8 > 9)
                         int a; 
                    else {
                        int a=9;}
                        
                int a = 90;
            }
            for(int a = 1, f =2;;)
                int a1=a;
            do{
            int a = 0;
            } while (!a != (1 + 9) * 2 / 8);
            
        }
    '''

s4 = '''
    void test3()
    {
    int s;
        int[] a = new int[5] {1,2,3,4,5};
        /*double b = 0.3;
        double[] fl = new double[] {1, b};*/ 
        a = new int[2]; 
        a[0];
        for (int i = 1; i < 3; i++) 
            a[i] = a[i-1]*2;
    }
    '''

s5 = '''
    void test4()
    {
        int a = 0, b = 1, c = 2, d = 3;
        a = b = c = d = -1;
        a = -(b + 2);
        c++;
        
    }
    '''

s6 = '''
    void t()
    {
        double a = 23 * 2 % 12 + -(10 / 4.0 + 2);
        bool b = !a && a || true && 92 != 0.0;
        bool c = 2 == 1.0;
    }
    '''

s7 = '''
    int c = 5;

    double func1(double a, int aa) {
      double r = 0.7 * a;
      r += r + c;
      return r;
    }
    double func1(double a, double aa) { return 1; }
    double func1(int a, int ada) { return 1; }
    double func1() { return 1;}
    int x = 0;
    double r = func1(7 + 2 + x, 1);
    double s = func1();
    '''

s8= '''
    // что-то с этим надо делать
    //int a, b = 0;
    void main()
    {
        string a = "hello";
        int aaa = 1+1;
        for (int i = 0; i<sizeof(a); i++)
        {
            output_char(a[i]);
        }
    }   
'''

test = '''
int a = 0;
int[] b = new int[] {0, 9};
    void main()
    {
        int a = input_int(), b = input_int();
        bool ab = !a && a>b || b-a;
        if (ab)
            output_int(5);
        if ( !(a==b) && a>b && 1 &&a)
           output_int(2);
        /*else
           output_int(3);
        if (a>b || !a || 0 || b)
           output_int(4);*/
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
test3 = '''
    /*int factorial(int n)
    {
        int s = 1;
        for (int i = 2; i <= n; i++)
            s *= i;
        return s;
    }
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

test4 = '''
    int[] a = new int[5];
    void sort()
    {
        for(int i=0;i<sizeof(a);i++)
            a[i] = 1;
    }
    void main()
    {
        for(int i=0;i<sizeof(a);i++)
            a[i] = input_int();
        sort();
        for(int i=0;i<sizeof(a);i++)
            output_int(a[i]);
    }
    
'''
build_tree(test4)
code_generate('test.c')
