#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<fstream>

using namespace std;

int main()
{
	srand(time(NULL));
	unsigned int i;
	printf("%u\n",i);
	ofstream fout;
	fout.open("random_dataset.txt");
	long long int j;
	for(j=0;j<1<<17;j++)
	{
		i = rand() % 65536;
		fout<<i<<endl;
	}
	i = rand() % 65536;
	fout<<i;
	fout.close();
	return 0;
}