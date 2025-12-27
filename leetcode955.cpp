class Solution {
public:
    int minDeletionSize(vector<string>& strs) {
        int cnt=0;

        int n =strs.size();
        int m =strs[0].size();
        bool f=false;
        bool equal=false;
        

        
        for(int j=0;j<m;j++){
            equal=false;
            for(int i=1;i<n;i++){
                
                if(strs[i-1][j]>strs[i][j]) {
                    cnt++;
                    f=true;
                    break;

                }
                else if(strs[i-1][j]==strs[i][j]) equal=true;
            }
            if(f==false && equal==false) return cnt;
        }
        return cnt;

        
        
    }
};