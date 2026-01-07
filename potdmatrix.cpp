class Solution {
public:
    long long maxMatrixSum(vector<vector<int>>& matrix) {
        long long  mini=INT_MAX;
        int cnt=0;
        

     long long  sum=0;
        int n=matrix.size();

        for(int i=0;i<n;i++){
            for(int j=0;j<n;j++){
                mini=min(mini,1LL*abs(matrix[i][j]));
                if(matrix[i][j]>0)sum+=matrix[i][j];
                else if(matrix[i][j]<0){
                 cnt++;
                 sum+=abs(matrix[i][j]);
                 }
                 
                 
            }
        }
        if(cnt%2==0) return sum;
        else return sum-2LL*mini;
    
    }
};