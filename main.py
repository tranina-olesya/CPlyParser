from cTreeParser import *
import os


s1 = '''
        bool eee, eee2 = false;
        int plus(int a, int b)
        {
            int res = a + b;
            return res;
        }
        float minus(float a, float b)
        {
            return a - b;
        }
        
        void main(float res)
        {
            c = input();
            for(int i = 0; i < 5; i++)
            {
                c = minus(plus(c, i), i);
                output(c);
                if ( c > 90)
                    return;
            }
        }
        ;
        void pass() { }
    '''

s2 = '''
        void test1() 
        {
            int g, g2 = g, g = 90;
    
            a = input(); b = input();  /* comment 1
            c = input();
            */
            for (int i = 0, j = 8; ((i <= 5)) && g; i = i + 1, print(5))
                for(; a < b;)
                    if (a > 7 + b) {
                        c = a + b * (2 - 1) + 0;  // comment 2
                        b = "98\tура";
                    }
                    else if (f)            
                        output(c + 1, 89.89);
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
    int c = 5;
    
    float f1(float a, float st) {
        int c = 1;
        
        float r = 7 * 2;
        r = r + c;
        return r;
    }
    
    int main()
    {
        int x;
        x = input();
        float r = f1(7 + 0.2, 70);
    }
    '''

#print(*build_tree(s7), sep=os.linesep)

print_tree(s7)