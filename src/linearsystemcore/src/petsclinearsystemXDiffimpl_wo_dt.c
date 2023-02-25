#include "petsclinearsystemXDiff.h"

static inline void fill_mat_values(PetscScalar *State, PetscInt i, PetscInt center, PetscInt j, PetscScalar *lowerLims, PetscScalar *upperLims, PetscScalar *dVec, PetscInt *incVec, PetscInt maxcols,PetscScalar *B, PetscScalar *C, PetscScalar dt, PetscInt *cols, PetscScalar *vals)
{ 
  PetscScalar firstCoefE = B[i],secondCoefE = C[i];

  //check whether it's at upper or lower boundary
  if (PetscAbs(State[i]-upperLims[j]) < dVec[j]/2.0) {//upper boundary
    vals[center] += -(firstCoefE/dVec[j] + secondCoefE/PetscPowReal(dVec[j],2));
    vals[center-1-2*j] = -(-firstCoefE/dVec[j] - 2.0*secondCoefE/PetscPowReal(dVec[j],2));
    vals[center-2-2*j] = -(secondCoefE/PetscPowReal(dVec[j],2));
    cols[center-1-2*j] = i - incVec[j];
    cols[center-2-2*j] = i - 2*incVec[j];
  } else if (PetscAbs(State[i]-lowerLims[j]) < dVec[j]/2.0) {//lower boundary
    vals[center] += -(-firstCoefE/dVec[j] + secondCoefE/PetscPowReal(dVec[j],2));
    vals[center+1+2*j] = -(firstCoefE/dVec[j] - 2.0*secondCoefE/PetscPowReal(dVec[j],2));
    vals[center+2+2*j] = -(secondCoefE/PetscPowReal(dVec[j],2));
    if (i + incVec[j] < maxcols) cols[center+1+2*j] = i + incVec[j]; // ignore out of bound entries
    if (i + 2*incVec[j] < maxcols) cols[center+2+2*j] = i + 2*incVec[j];
  } else {
    //first derivative
    vals[center] += -(-firstCoefE*(firstCoefE > 0) + firstCoefE*( firstCoefE<0))/dVec[j] - (-2)*secondCoefE/(PetscPowReal(dVec[j],2));
    vals[center+1+2*j] = -firstCoefE*(firstCoefE > 0)/dVec[j] - secondCoefE/(PetscPowReal(dVec[j],2));
    vals[center-1-2*j] = -firstCoefE*(firstCoefE < 0)/dVec[j] - secondCoefE/(PetscPowReal(dVec[j],2));
    cols[center-1-2*j] = i - incVec[j];
    if (i + incVec[j] < maxcols) cols[center+1+2*j] = i + incVec[j]; // ignore out of bound entries
  }
}

static inline void fill_mat_values_DirectDiff(PetscScalar *State, PetscInt i, PetscInt center, PetscInt j, PetscScalar *lowerLims, PetscScalar *upperLims, PetscScalar *dVec, PetscInt *incVec, PetscInt maxcols,PetscScalar *B, PetscScalar *C, PetscScalar dt, PetscInt *cols, PetscScalar *vals)
{ 
  PetscScalar firstCoefE = B[i],secondCoefE = C[i];
  PetscScalar dVecX = dVec[j], dVecX_sq = PetscPowReal(dVec[j],2) ;

  //check whether it's at upper or lower boundary
  if (PetscAbs(State[i]-upperLims[j]) < dVecX/2.0) {
    //upper boundary
    //v_{N+1} = v_{N}, ghost point N+1 

    vals[center] += -(   (    -firstCoefE*(firstCoefE > 0) + firstCoefE*( firstCoefE<0) )/ dVecX - 2* secondCoefE/dVecX_sq);  // coef of v_{N}
    vals[center] += -(firstCoefE*(firstCoefE > 0)/dVecX + secondCoefE/dVecX_sq); // add coef of v_{N+1}

    vals[center-1-2*j] = -(  -firstCoefE*(firstCoefE < 0)/dVecX + secondCoefE/dVecX_sq );

    cols[center-1-2*j] = i - incVec[j];



  } else if (PetscAbs(State[i]-lowerLims[j]) < dVecX/2.0) {
    //lower boundary
    //v_{0} = v_{1}, ghost point N+1 

    vals[center] += -(   (  -firstCoefE*(firstCoefE > 0) + firstCoefE*( firstCoefE<0) )/ dVecX - 2* secondCoefE/dVecX_sq);  // coef of v_{1}
    vals[center] += -(-firstCoefE*(firstCoefE < 0)/dVecX + secondCoefE/dVecX_sq); // add coef of v_{0}
    vals[center+1+j] = -(  firstCoefE*(firstCoefE > 0) /dVecX + secondCoefE/dVecX_sq  );

    if (i + incVec[j] < maxcols) cols[center+1+j] = i + incVec[j]; // ignore out of bound entries

  } else {
    //interior points
    vals[center] += -(   (  -firstCoefE*(firstCoefE > 0) + firstCoefE*( firstCoefE<0) )/ dVecX - 2* secondCoefE/dVecX_sq);  // coef of v_{i}
    vals[center+1+j] = -(  firstCoefE*(firstCoefE > 0) /dVecX + secondCoefE/dVecX_sq  ); // coef of v_{i+1}
    vals[center-1-j] = -(  -firstCoefE*(firstCoefE < 0)/dVecX + secondCoefE/dVecX_sq ); // coef of v_{i-1}

    cols[center-1-j] = i - incVec[j];
    if (i + incVec[j] < maxcols) cols[center+1+j] = i + incVec[j]; // ignore out of bound entries

  }

  
}

static inline void fill_mat_values_CrossDiff(PetscScalar *StateX, PetscScalar *StateY, PetscInt i, PetscInt center, PetscInt jX, PetscInt jY, PetscScalar *lowerLims, PetscScalar *upperLims, PetscScalar *dVec, PetscInt *incVec, PetscInt maxcols, PetscScalar *C_XY, PetscScalar dt, PetscInt *cols, PetscScalar *vals)
{ 
  PetscScalar crossCoefE = C_XY[i];
  PetscScalar dVecX = dVec[jX];
  PetscScalar dVecY = dVec[jY];

  //check three cases
  // case 1: corner point
  // case 2: edge line
  // case 3: interior


/************************************************/
  // case 1: corner point
/************************************************/

  if (  PetscAbs(StateX[i]-upperLims[jX]) < dVecX/2.0 && PetscAbs(StateY[i]-upperLims[jY]) < dVecY /2.0 )  {
  // case 1: corner point  // top right 

    vals[center] += -(  crossCoefE/(4*dVecX*dVecY) ); 
    vals[center-2-4*jX] += -(  -crossCoefE/(4*dVecX*dVecY) );
    vals[center-4-4*jX] += -(  -crossCoefE/(4*dVecX*dVecY) );

    vals[center+4+4*jX] += -(  crossCoefE/(4*dVecX*dVecY) );
    
    cols[center-2-4*jX] = i - incVec[jX];
    cols[center-4-4*jX] = i - incVec[jY];
    cols[center+4+4*jX] = i - incVec[jX]-incVec[jY];


  } else if (  PetscAbs(StateX[i]-upperLims[jX]) < dVecX/2.0 && PetscAbs(StateY[i]-lowerLims[jY]) < dVecY /2.0 ) {
  // case 1: corner point  // bottom right 

    vals[center] += -(  -crossCoefE/(4*dVecX*dVecY) ); 
    vals[center-1-4*jX] += -(  crossCoefE/(4*dVecX*dVecY) );
    vals[center-2-4*jX] += -(  crossCoefE/(4*dVecX*dVecY) );

    vals[center+1+4*jX] += -(  -crossCoefE/(4*dVecX*dVecY) );
    
    cols[center-1-4*jX] = i + incVec[jY];
    cols[center-2-4*jX] = i - incVec[jX];
    cols[center+1+4*jX] = i - incVec[jX]+incVec[jY];



  } else if (  PetscAbs(StateX[i]-lowerLims[jX]) < dVecX/2.0 && PetscAbs(StateY[i]-upperLims[jY]) < dVecY /2.0 ) {
  // case 1: corner point  // top left 

    vals[center] += -(  -crossCoefE/(4*dVecX*dVecY) ); 
    vals[center-3-4*jX] += -(  crossCoefE/(4*dVecX*dVecY) );
    vals[center-4-4*jX] += -(  crossCoefE/(4*dVecX*dVecY) );

    vals[center+3+4*jX] += -(  -crossCoefE/(4*dVecX*dVecY) );
    
    cols[center-3-4*jX] = i + incVec[jX];
    cols[center-4-4*jX] = i - incVec[jY];
    cols[center+3+4*jX] = i + incVec[jX]-incVec[jY];



  } else if (  PetscAbs(StateX[i]-lowerLims[jX]) < dVecX/2.0 && PetscAbs(StateY[i]-lowerLims[jY]) < dVecY /2.0 ) {
  // case 1: corner point  // bottom left 

    vals[center] += -(  crossCoefE/(4*dVecX*dVecY) ); 
    vals[center-1-4*jX] += -(  -crossCoefE/(4*dVecX*dVecY) );
    vals[center-3-4*jX] += -(  -crossCoefE/(4*dVecX*dVecY) );

    vals[center+2+4*jX] += -(  crossCoefE/(4*dVecX*dVecY) );
    
    cols[center-1-4*jX] = i + incVec[jY];
    cols[center-3-4*jX] = i + incVec[jX];
    cols[center+2+4*jX] = i +incVec[jY] + incVec[jX] ;

  }
  


/************************************************/
  // case 2: edge line
/************************************************/

  // if (  PetscAbs(StateX[i]-lowerLims[jX]) < dVecX/2.0  && PetscAbs(StateX[i]-upperLims[jX]) < dVecX/2.0 && PetscAbs(StateY[i]-upperLims[jY]) < dVecY /2.0 )  {
  if (  PetscAbs(StateX[i]-lowerLims[jX]) > dVecX/2.0  && PetscAbs(StateX[i]-upperLims[jX]) > dVecX/2.0 && PetscAbs(StateY[i]-upperLims[jY]) < dVecY /2.0 )  {
  // case 2: top


    vals[center-2-4*jX] += -(  -crossCoefE/(4*dVecX*dVecY) );
    vals[center-3-4*jX] += -(  crossCoefE/(4*dVecX*dVecY) );

    vals[center+3+4*jX] += -(  -crossCoefE/(4*dVecX*dVecY) );
    vals[center+4+4*jX] += -(  crossCoefE/(4*dVecX*dVecY) );
    
    cols[center-2-4*jX] = i - incVec[jX];
    cols[center-3-4*jX] = i + incVec[jX];
    cols[center+3+4*jX] = i + incVec[jX]-incVec[jY];
    cols[center+4+4*jX] = i - incVec[jX]-incVec[jY];


  // } else if (  PetscAbs(StateX[i]-lowerLims[jX]) < dVecX/2.0  && PetscAbs(StateX[i]-upperLims[jX]) < dVecX/2.0 && PetscAbs(StateY[i]-lowerLims[jY]) < dVecY /2.0 )  {
  } else if (  PetscAbs(StateX[i]-lowerLims[jX]) > dVecX/2.0  && PetscAbs(StateX[i]-upperLims[jX]) > dVecX/2.0 && PetscAbs(StateY[i]-lowerLims[jY]) < dVecY /2.0 )  {
  // case 2: bottom

    vals[center-2-4*jX] += -(  crossCoefE/(4*dVecX*dVecY) );
    vals[center-3-4*jX] += -(  -crossCoefE/(4*dVecX*dVecY) );

    vals[center+1+4*jX] += -(  -crossCoefE/(4*dVecX*dVecY) );
    vals[center+2+4*jX] += -(  crossCoefE/(4*dVecX*dVecY) );
    
    cols[center-2-4*jX] = i - incVec[jX];
    // cols[center-3-4*jX] = i - incVec[jY];
    cols[center-3-4*jX] = i + incVec[jX];
    cols[center+1+4*jX] = i - incVec[jX]+incVec[jY];
    cols[center+2+4*jX] = i + incVec[jX]+incVec[jY];



  // } else if (  PetscAbs(StateX[i]-lowerLims[jX]) < dVecX/2.0 && PetscAbs(StateY[i]-upperLims[jY]) < dVecY /2.0  && PetscAbs(StateY[i]-lowerLims[jY]) < dVecY /2.0) {
  } else if (  PetscAbs(StateX[i]-lowerLims[jX]) < dVecX/2.0 && PetscAbs(StateY[i]-upperLims[jY]) > dVecY /2.0  && PetscAbs(StateY[i]-lowerLims[jY]) > dVecY /2.0) {
  // case 2: left

    vals[center-1-4*jX] += -(  -crossCoefE/(4*dVecX*dVecY) );
    vals[center-4-4*jX] += -(  crossCoefE/(4*dVecX*dVecY) );

    vals[center+2+4*jX] += -(  crossCoefE/(4*dVecX*dVecY) );
    vals[center+3+4*jX] += -(  -crossCoefE/(4*dVecX*dVecY) );
    
    cols[center-1-4*jX] = i + incVec[jY];
    cols[center-4-4*jX] = i - incVec[jY];
    cols[center+2+4*jX] = i + incVec[jX]+incVec[jY];
    cols[center+3+4*jX] = i + incVec[jX]-incVec[jY];

  // } else if (  PetscAbs(StateX[i]-upperLims[jX]) < dVecX/2.0 && PetscAbs(StateY[i]-upperLims[jY]) < dVecY /2.0  && PetscAbs(StateY[i]-lowerLims[jY]) < dVecY /2.0) {
  } else if (  PetscAbs(StateX[i]-upperLims[jX]) < dVecX/2.0 && PetscAbs(StateY[i]-upperLims[jY]) > dVecY /2.0  && PetscAbs(StateY[i]-lowerLims[jY]) > dVecY /2.0) {
  // case 2: right

    vals[center-1-4*jX] += -(  crossCoefE/(4*dVecX*dVecY) );
    vals[center-4-4*jX] += -(  -crossCoefE/(4*dVecX*dVecY) );

    vals[center+1+4*jX] += -(  -crossCoefE/(4*dVecX*dVecY) );
    vals[center+4+4*jX] += -(  crossCoefE/(4*dVecX*dVecY) );
    
    cols[center-1-4*jX] = i + incVec[jY];
    cols[center-4-4*jX] = i - incVec[jY];
    cols[center+1+4*jX] = i - incVec[jX]+incVec[jY];
    // cols[center+2+4*jX] = i + incVec[jX]+incVec[jY];
    cols[center+4+4*jX] = i - incVec[jX]-incVec[jY];

  }
  
/************************************************/
  // case 3: interior
/************************************************/


  // if (  PetscAbs(StateX[i]-lowerLims[jX]) < dVecX/2.0  && PetscAbs(StateX[i]-upperLims[jX]) < dVecX/2.0 && PetscAbs(StateY[i]-upperLims[jY]) < dVecY /2.0 &&PetscAbs(StateY[i]-lowerLims[jY]) < dVecY /2.0 )  {
  if (  PetscAbs(StateX[i]-lowerLims[jX]) > dVecX/2.0  && PetscAbs(StateX[i]-upperLims[jX]) > dVecX/2.0 && PetscAbs(StateY[i]-upperLims[jY]) > dVecY /2.0 &&PetscAbs(StateY[i]-lowerLims[jY]) > dVecY /2.0 )  {

    vals[center+1+4*jX] = -(  -crossCoefE/(4*dVecX*dVecY) );
    vals[center+2+4*jX] = -(  crossCoefE/(4*dVecX*dVecY) );
    vals[center+3+4*jX] = -(  -crossCoefE/(4*dVecX*dVecY) );
    vals[center+4+4*jX] = -(  crossCoefE/(4*dVecX*dVecY) );
    
    cols[center+1+4*jX] = i - incVec[jX] + incVec[jY] ;
    cols[center+2+4*jX] = i + incVec[jX] + incVec[jY];
    cols[center+3+4*jX] = i + incVec[jX] - incVec[jY];
    cols[center+4+4*jX] = i - incVec[jX] - incVec[jY];


  }

}



PetscErrorCode FormLinearSystem_C(PetscScalar *R, PetscScalar *F, PetscScalar *K, PetscScalar *A, PetscScalar *B_r, PetscScalar *B_f, PetscScalar *B_k, PetscScalar *C_rr, PetscScalar *C_ff, PetscScalar *C_kk, PetscScalar dt, PetscScalar *lowerLims, PetscScalar *upperLims, PetscScalar *dVec, PetscInt *incVec, PetscInt n, Mat petsc_mat)
{
  PetscErrorCode ierr;
  PetscInt       i, center;
  PetscInt       cols[13];
  PetscScalar    vals[13];

  PetscFunctionBegin;
  for (i = 0; i < n; ++i) {
    center = 3*4/2;
    memset(vals,0,13*sizeof(PetscScalar));
    memset(cols,-1,13*sizeof(PetscInt));
    cols[center] = i;
    vals[center] = 1.0 - dt * A[i];
    fill_mat_values(R,i,center,0,lowerLims,upperLims,dVec,incVec,n,B_r,C_rr,dt,cols,vals);
    fill_mat_values(F,i,center,1,lowerLims,upperLims,dVec,incVec,n,B_f,C_ff,dt,cols,vals);
    fill_mat_values(K,i,center,2,lowerLims,upperLims,dVec,incVec,n,B_k,C_kk,dt,cols,vals);
    ierr = MatSetValues(petsc_mat,1,&i,3*4+1,cols,vals,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(petsc_mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(petsc_mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



PetscErrorCode FormLinearSystem_DirectCrossDiff_C(PetscScalar *R, PetscScalar *F, PetscScalar *K, PetscScalar *A, PetscScalar *B_r, PetscScalar *B_f, PetscScalar *B_k, PetscScalar *C_rr, PetscScalar *C_ff, PetscScalar *C_kk, PetscScalar *C_rf, PetscScalar *C_fk, PetscScalar *C_kr, PetscScalar dt, PetscScalar *lowerLims, PetscScalar *upperLims, PetscScalar *dVec, PetscInt *incVec, PetscInt n, Mat petsc_mat)
{
  PetscErrorCode ierr;
  PetscInt       i, center, centerXDiff;
  PetscInt       cols[13];
  PetscScalar    vals[13];

  PetscInt       colsXDiff[25];
  PetscScalar    valsXDiff[25];


  PetscFunctionBegin;
  for (i = 0; i < n; ++i) {
    center = 3*4/2;
    memset(vals,0,13*sizeof(PetscScalar));
    memset(cols,-1,13*sizeof(PetscInt));
    cols[center] = i;
    vals[center] = 1.0/dt - A[i];
    fill_mat_values(R,i,center,0,lowerLims,upperLims,dVec,incVec,n,B_r,C_rr,dt,cols,vals);
    fill_mat_values(F,i,center,1,lowerLims,upperLims,dVec,incVec,n,B_f,C_ff,dt,cols,vals);
    fill_mat_values(K,i,center,2,lowerLims,upperLims,dVec,incVec,n,B_k,C_kk,dt,cols,vals);

    PetscCall(MatSetValues(petsc_mat,1,&i,3*4+1,cols,vals,INSERT_VALUES));
  }

    PetscCall(MatAssemblyBegin(petsc_mat,MAT_FLUSH_ASSEMBLY));
    PetscCall(MatAssemblyEnd(petsc_mat,MAT_FLUSH_ASSEMBLY));

    for (i = 0; i < n; ++i) {

    centerXDiff = 3*8/2;
    memset(valsXDiff,0,25*sizeof(PetscScalar));
    memset(colsXDiff,-1,25*sizeof(PetscInt));
    colsXDiff[center] = i;
    valsXDiff[center] = 1.0/dt -  A[i];

    fill_mat_values_CrossDiff(R,F, i,centerXDiff,0, 1 ,lowerLims,upperLims,dVec,incVec,n,C_rf,dt,colsXDiff,valsXDiff);
    fill_mat_values_CrossDiff(F,K, i,centerXDiff,1, 2 ,lowerLims,upperLims,dVec,incVec,n,C_fk,dt,colsXDiff,valsXDiff);
    // fill_mat_values_CrossDiff(K,R, i,centerXDiff,2, 0 ,lowerLims,upperLims,dVec,incVec,n,C_kr,dt,colsXDiff,valsXDiff);
    PetscCall(MatSetValues(petsc_mat,1,&i,3*8+1,colsXDiff,valsXDiff,ADD_VALUES));

  }

  ierr = MatAssemblyBegin(petsc_mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(petsc_mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

