from cTreeParser import *
import os


s1 = '''
        int plus(int a, int b)
        {
            int res = a + b;
            return res;
        }
        float minus(float a, float b)
        {
            return a - b;
        }
        
        int main(float res)
        {
            float c = 0;
            for(int i = 0; i < 5; i++)
            {
                c = minus(plus(1, i), i);
                if ( c > 90)
                    return;
            }
        }
        ;
        int pass() { }
    '''

s2 = '''
        int a, b;
        int test1() 
        {
            int g, g2 = g;
    
            /* comment 1
            c = input();
            */
            for (int i = 0, j = 8; ((i <= 5)) && g; i = i + 1)
                for(; a < b;)
                    if (a > 7 + b) {
                        int c = a + b * (2 - 1) + 0;  // comment 2
                        str ab = "98\tура";
                    }
                    else if (a > 4)   
                        ab = ab + "a";         
                        //output(c + 1, 89.89);
            for(;;);
        }
    '''

s3 = '''
        void test2() 
        {
            while (a==0){
                if (a > b + 1 && x) 
                    a = 0;
                else 
                    if (8 > 9)
                         output(a); 
                    else {
                        a=9;}
                        
                a = 90;
            }
            
            do{
            } while (!a != (1 + 9) * b / 8);
            
        }
    '''

s4 = '''
    void test3()
    {
        int res = 0, s;
        int[] a = new int[] {1,2,3,4,5};
        float b = 0.3;
        float[] fl = new int[2] {1.2, b}; 
        
        for (int i = 1; i < len(a); i++) 
            a[i] = a[i-1]*2;
    }
    '''

s5 = '''
    void test4()
    {
        int a = 0, b = 1, c = 2, d = 3;
        a = b = c = d;
        a = b + -2;
        a++;
    }
    '''

s6 = '''
    int t()
    {
        int[] a = new int[5], b = new int[] {1,2+0, a}, c = new int[1] {1}, d;
        a[9] = int(a[o+1]);
    }
    '''

s7 = '''
    float c = 2.0 + -5;
    
    int func1(float a) {
      float r = 0.7 * a;
      r += r + c;
      return r;
    }
    
    float a = 2 && 0 || !1.0;
    //float x = -(81 % (3 + -1)) && 2.0;
    //float r = func1(7 + 0.2 + x);
    '''

#print(*build_tree(s7), sep=os.linesep)

print_tree(s7)