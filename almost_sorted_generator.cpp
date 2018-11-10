#include<bits/stdc++.h>
#include<fstream>

using namespace std;

int main()
{
	vector <unsigned int> v;
	srand(time(NULL));
	long long int i,x,y;
	unsigned int temp;
	for(i=0;i<1000000;i++)
	{
		v.push_back(rand() % 65536);
	}
	sort(v.begin(),v.end());
	for(i=0;i<0.1*1000000;i++)
	{
		x = rand() % (int)(1000000);
		y = rand() % (int)(1000000);
		temp = v[x];
		v[x]=v[y];
		v[y]=temp;
	}
	ofstream fout;
	fout.open("almost_sorted_dataset.txt");
	for(i=0;i<999999;i++)
	{
		fout<<v[i]<<endl;
	}
	fout<<v[i];
	fout.close();
	return 0;
}