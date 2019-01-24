int[] a = new int[10]; 
int size = 10;
void between_zeros()
{
	int i = 0;
	bool first = false, last = false;
	
	while(i < size && !last)
	{
		if (a[i]==0)
		{	
			if (!first)
				first = 1;
			else
				last = 1;
		}
		else if (first)
			output(a[i]);
		i++;
	}
}

int main() { 
	for (int i =0; i<size;i++)
		a[i] = input_int();
	output(' ');
	between_zeros();
	return 0;
}
