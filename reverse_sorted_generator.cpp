#include<bits/stdc++.h>
#include<fstream>

using namespace std;

int main()
{
	vector <unsigned int> v;
	srand(time(NULL));
	long long int i;
	for(i=0;i<1000000;i++)
	{
		v.push_back(rand() % 65536);
	}
	sort(v.begin(),v.end(),greater<unsigned int>());
	ofstream fout;
	fout.open("reverse_dataset.txt");
	for(i=0;i<999999;i++)
	{
		fout<<v[i]<<endl;
	}
	fout<<v[i];
	fout.close();
	return 0;
}