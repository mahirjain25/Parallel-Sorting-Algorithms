#include<bits/stdc++.h>
#include<fstream>

using namespace std;

int main(int argc, char*argv[])
{
	vector <unsigned int> v;
	srand(time(NULL));
	long long int i,x,y,num;
	unsigned int temp;
	num = (long long int)atoll(argv[1]);
	for(i=0;i<num;i++)
	{
		v.push_back(rand() % 65536);
	}
	sort(v.begin(),v.end());
	for(i=0;i<(long long int)0.1*num;i++)
	{
		x = rand() % (int)(1000000);
		y = rand() % (int)(1000000);
		temp = v[x];
		v[x]=v[y];
		v[y]=temp;
	}
	ofstream fout;
	fout.open("almost_sorted_dataset.txt");
	for(i=0;i<num-1;i++)
	{
		fout<<v[i]<<endl;
	}
	fout<<v[i];
	fout.close();
	return 0;
}