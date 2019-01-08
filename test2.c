int[] a;
void main() { 
	a = new int[4];
	for(int i = 9; i < 9+ sizeof(a)/sizeof(a[0]); i++)
		a[i-9] = i;
	
	for(int i = 0; i<sizeof(a)/sizeof(a[0]); i++)
		output(a[i]);
	a = new int[] {19,3};
	for(int i = 0; i<sizeof(a)/sizeof(a[0]); i++)
		output(a[i]);
	int[] a = new int[] {a[1], 0, a[0]};
	for(int i = 0; i<sizeof(a)/sizeof(a[0]); i++)
		output(a[i]);
}