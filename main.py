from c_tree_parser import *

test4 = '''

int f(int n)
{
	if (n == 1 || n == 2)
	{
		return 1;
		int a = 0;
		output(a);
	}
	return f(n - 1) + f(n - 2);
}
void fibonacci()
{
	output("fibonacci:");
	int n = input_int();
	output(' ');
	for (int i = 1; i <= n; i++)
		output(f(i));
}

int factorial(int n)
{
	int s = 1;
	for (int i = 1; i <= n; i++)
		s *= i;
	return s;
}

double[] a = new double[] { 1,2.2,35,6,33,5.0,6.2,0,8 };
int search(double el)
{
	int i, n = sizeof(a);
	for (i = 0; i < n; i++)
		if (a[i] == el)
			break;
	if (i == n)
		return -1;
	else
		return i;
}

void bool_values()
{
	int a = input_int(), b = input_int();
	output(' ');
	bool ab = !a && a > b || b - a;
	if (ab)
		output(1);
	if (!(a > b) && a >= b && 1 && a)
		output(2);
	else
		output(3);
	if (a > b || !a || 0 || b)
		output(4);
}

void sort()
{
	int[] a = new int[10];
	int size = 10;
	for (int i = 0; i < size; i++)
		a[i] = input_int();

	for (int i = 0; i < size - 1; i++) {
		for (int j = 0; j < size - i - 1; j++) {
			if (a[j] > a[j + 1]) {
				int temp = a[j];
				a[j] = a[j + 1];
				a[j + 1] = temp;
			}
		}
	}
	output(' ');
	for (int i = 0; i < size; i++)
		output(a[i]);
}

void waiting_for_100()
{
	int x = 37;
	do {
		int x = input_int();
		if (x < 0)
			continue;
		output(x);
	} while (x != 100);
	output(' ');
	output(x);
}

int main()
{
			sort();
	return 0;
}

'''
build_tree(test4)
# code_generate(sys.argv[1])
