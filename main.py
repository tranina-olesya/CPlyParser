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
            {
                if(minus(a) > 4)
                    return 0;
                else
                    return minus(plus(1, 9));
            }
        }
        ;
        void pass() { }
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
                    else if (2 > 4)   
                        ab += "a";         
                        //output(c + 1, 89.89);
            for(;;);
        }
    '''

s3 = '''
        void test2() 
        {
            int a;
            while (a==0){
                if (a > 23 + 1 && 2) 
                    int a = 0;
                else 
                    if (8 > 9)
                         int a; 
                    else {
                        int a=9;}
                        
                int a = 90;
                //trtet();
            }
            for(int a = 1;;)
                int a1=a;
            do{
            int a = 0;
            } while (!a != (1 + 9) * 2 / 8);
            
        }
    '''

s4 = '''
    void test3()
    {
        int res = 0, s;
        int[] a = new int[5] {1,2,3,4,5};
        double b = 0.3;
        double[] fl = new double[] {1.2, b}; 
        int[] aw = new int[2*(0+4)+13%5], d;
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
        a++;
        
    }
    '''

s6 = '''
    void t()
    {
        double a = 23 * 2 % 12 + -(10 / 4.0 + 2);
        bool b = !a && !"" || true && 92 != 0.0;
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
    void main()
    {
        string a = "aaa";
        a[1];
    }
'''

test = '''
    void main()
    {
        int n = input_int();
        int s = 1;
        for (int i = 2; i <= n; i++)
            s *= i;
        output_ints(s);
    }

'''

test2 = '''
    void main()
    {
        int n = 6;
        int a = 1;
        int c = a = n = 9;
    }
'''
print_tree(test2)
