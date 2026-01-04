class Solution {
public:
    long long maximumHappinessSum(vector<int>& happiness, int k) {
        long long ansI=0;
        int y=0;
        sort(happiness.begin(),happiness.end());

        for(int i=happiness.size()-1;i>=0;i--){
            if(y<k){
                int x=happiness[i]-y;
                if(x<0) x=0;
                ans+=x;
                y++;
            }

        }
        return ans;
    }
};
