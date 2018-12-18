module chamber(chamberSize = [36,25,10], grid = [6,4], griddW = 1, edgedW = 4, height = 10){
    
    geneFrameL = chamberSize[0];
    geneFrameW = chamberSize[1];
    height = chamberSize[2];

    
    nX = grid[0];
    nY = grid[1];
    
    wellL = (geneFrameL - 2*edgedW - (nX + 1) * griddW) / nX;
    wellW = (geneFrameW - 2*edgedW - (nY + 1) * griddW) / nY;
    
    echo(wellL, wellW);
    
    difference(){
        cube([geneFrameL, geneFrameW, height]);
        for(i = [0:nY-1]){
            dY = griddW + edgedW + i * ( griddW + wellW );
            for(j = [0:nX-1]){
                dX = griddW + edgedW + j * ( griddW + wellL );
                translate([dX, dY]) cube([wellL, wellW, height]);
            }
        }  
    }
    cube([geneFrameL, geneFrameW, 1]);
}

chamber();