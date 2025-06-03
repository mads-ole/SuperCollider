//=== ROUTINE FOR CONROLLING THE SOUND MODULATIONS ===\\
// based on brightness tracking data from python

// ---------------------------------------------------------------- *
// FUNCTION FOR MAPPING THE 2D-PLAYFIELD/MATRIX POSITIONS ON A HALF SPHERE
(
~car2pol = {
	arg gridSize, row, col;
	var mid, azimuth, elevation, gridPoint, distance, volume;
	mid = (gridSize - 1) / 2;
	// Math magic
	gridPoint = Point.new(mid- row,mid - col).asPolar;
	gridPoint = gridPoint;
	azimuth = gridPoint.theta;
	elevation = 2 * (1/gridPoint.rho).atan;
	azimuth = azimuth * 180 / pi;
	elevation = elevation * 90 / pi;
	[azimuth, elevation];
};
)

// ---------------------------------------------------------------- *
// CREATE LIST OF BOAT POSITIONINGS THE PLAYERS HAVE CHOOSEN

(//boat positions player 1
var boat1, boat2, boat3, boat4, boat5;
boat1 = [[2, 0], [2, 1], [2, 2]];
boat2 = [[3, 2], [3, 3], [3, 4]];
boat3 = [[0, 1], [0, 2]];
boat4 = [[0, 4], [1, 4]];
boat5 = [[4, 1]];

~boats1 = [boat1, boat2, boat3, boat4, boat5];
)

(//boat positions player 2
var boat1, boat2, boat3, boat4, boat5;
boat1 = [[1, 2], [1, 3], [1, 4]];
boat2 = [[2, 4], [3, 4], [4, 4]];
boat3 = [[2, 2], [3, 2]];
boat4 = [[4, 0], [4, 1]];
boat5 = [[1, 0]];

~boats2 = [boat1, boat2, boat3, boat4, boat5];
)

// ---------------------------------------------------------------- *
// ROUTINE FOR MATRIX 1
// Matrix1/player1's OSC-data (~array2D_1) will be compared to boat positions of player2 (~boats2)
// initial state of ~array2D_1 should be all zeros

Tdef(\matrix_1).play;
Tdef(\matrix_1).stop;
(
var defList = [Ndef(\src1_1), Ndef(\src2_1), Ndef(\src3_1), Ndef(\src4_1), Ndef(\src5_1)], //rows (sources)
modList = [\mod1, \mod2, \mod3, \mod4, \mod5], //columns (modulations)
ampList = [\src1, \src2, \src3, \src4, \src5]; //source amplification

Tdef(\matrix_1, {
	var oldInd = 0, row, col, arr_old = ~array2D_1;
	loop {
		2.0.wait;
		"step".postln;

		//iterate rows
		~array2D_1.rowsDo({
			arg item, ind; //ind = rownumber

			//iterate columns
			item.do({
				arg item2, ind2; //ind2 = columnnumber


				if (item2 != arr_old[ind, ind2],
				//"if status of field at position [ind, ind2] changed:"

					//true:
					{
						//umute Ndef in row ind
						//mute Ndef that got played before - only one sound per matrix should play
						Ndef(\mix_1).set(ampList[oldInd], 0.0);
						Ndef(\mix_1).set(ampList[ind], 0.9);
						oldInd = ind;

						//set spatialization/azimut&elelevation of playing Ndef
						row = ind;
						col = ind2;
						defList[ind].set(\azi, ~car2pol.(5, row, col)[0]);
						defList[ind].set(\ele, ~car2pol.(5, row, col)[1]);

						//set the modulation
						if (item2 == 1,
							//"if field got covered:"

							//true
							{
								defList[ind].set(modList[ind2], 1);
							},

							//"if field got uncovered:"
							//false
							{
								defList[ind].set(modList[ind2], 0);
							}
						);

						if (item.sum == 0,
						//"if there is no modulation on item(=Ndef):"

							//true
							{
								Ndef(\mix_1).set(ampList[ind], 0.0);
							}
						);
					}
				);
			});
		});

		arr_old = ~array2D_1.deepCopy;

		//=======================================
		//hit tracking (matrix1 vs. boats2)

		~boats2.do({
			arg boat, ind; //boat = item

			boat.do({
				arg item2, ind2; //item2 = coordinate

				if(item2 == [row, col],
				//"if item2 corresponds to played position from above(row, col)"
				//-> boat got hit

					//true
					{
						"treffer".postln;
						boat[ind2] = 0;

						//Corresponding Ndef/sound spatialization rotates one time
						defList[row].set(\azi, (~car2pol.(5, row, col)[0]) + 360);

				});

				if(boat.sum == 0,
				//"if all coordinates of a boat got played:"

					//true
					{
						"schiff versenkt".postln;
						~boats2.removeAt(ind);

						//all sounds turn on once and get spatialized throug the room
						fork{
							1.do({
								Ndef(\mix_1).set(ampList[0], 0.9);
								Ndef(\mix_1).set(ampList[1], 0.9);
								Ndef(\mix_1).set(ampList[2], 0.9);
								Ndef(\mix_1).set(ampList[3], 0.9);
								Ndef(\mix_1).set(ampList[4], 0.9);
								defList[0].set(\azi, ((340.rand) - 170));
								defList[1].set(\azi, ((340.rand) - 170));
								defList[2].set(\azi, ((340.rand) - 170));
								defList[3].set(\azi, ((340.rand) - 170));
								defList[4].set(\azi, ((340.rand) - 170));
								5.0.wait;
								Ndef(\mix_1).set(ampList[0], 0.0);
								Ndef(\mix_1).set(ampList[1], 0.0);
								Ndef(\mix_1).set(ampList[2], 0.0);
								Ndef(\mix_1).set(ampList[3], 0.0);
								Ndef(\mix_1).set(ampList[4], 0.0);

								//only Ndef of the last played row remains
								Ndef(\mix_1).set(ampList[row], 0.9);
							});
						};
				});


				if(~boats2.sum == 0,
				//"if all boats been hit:"

					//true
					{
						"field 1 won".postln;


						fork{
							1.do({
								Tdef(\matrix_2).stop;

								//all sounds turn on and circle throug the room
								Ndef(\mix_1).set(ampList[0], 0.9);
								Ndef(\mix_1).set(ampList[1], 0.9);
								Ndef(\mix_1).set(ampList[2], 0.9);
								Ndef(\mix_1).set(ampList[3], 0.9);
								Ndef(\mix_1).set(ampList[4], 0.9);
								defList[0].set(\azi, ((340.rand) - 170));
								defList[1].set(\azi, ((340.rand) - 170));
								defList[2].set(\azi, ((340.rand) - 170));
								defList[3].set(\azi, ((340.rand) - 170));
								defList[4].set(\azi, ((340.rand) - 170));
								5.0.wait;
								defList[0].set(\azi, ((340.rand) - 170));
								defList[1].set(\azi, ((340.rand) - 170));
								defList[2].set(\azi, ((340.rand) - 170));
								defList[3].set(\azi, ((340.rand) - 170));
								defList[4].set(\azi, ((340.rand) - 170));

								defList[0].set(\ele, Rand.new(u, 90));
								defList[1].set(\ele, Rand.new(u, 90));
								defList[2].set(\ele, Rand.new(u, 90));
								defList[3].set(\ele, Rand.new(u, 90));
								defList[4].set(\ele, Rand.new(u, 90));
								5.0.wait;

								//all sound from Matrix2 turn on aswell
								Ndef(\mix_2).set(ampList[0], 0.9);
								Ndef(\mix_2).set(ampList[1], 0.9);
								Ndef(\mix_2).set(ampList[2], 0.9);
								Ndef(\mix_2).set(ampList[3], 0.9);
								Ndef(\mix_2).set(ampList[4], 0.9);
								5.0.wait;

								//Matrix1 sounds get turned off
								Ndef(\mix_1).set(\amp, 0.0);

								//"extra Defs" stop playing
								7.0.wait;
								Ndef(\lowpad).stop(2);
								Ndef(\sprinkle).stop(2);
								Ndef(\bassy).stop(2);
								Ndef(\pad).stop(2);
								Ndef(\texture).stop(2);
								2.0.wait;

								//Matrix2 sounds get turned off
								Ndef(\mix_2).set(ampList[0], 0.0);
								Ndef(\mix_2).set(ampList[1], 0.0);
								Ndef(\mix_2).set(ampList[2], 0.0);
								Ndef(\mix_2).set(ampList[3], 0.0);
								Ndef(\mix_2).set(ampList[4], 0.0);
						});
					};
					//this Routine gets stopped
					Tdef(\matrix_1).stop;
				});
			});
		});
	}
});
)

// ---------------------------------------------------------------- *
//ROUTINE FOR MATRIX2
//same as above but for the second playing field

Tdef(\matrix_2).play;
Tdef(\matrix_2).stop;
(
var defList = [Ndef(\src1_2), Ndef(\src2_2), Ndef(\src3_2), Ndef(\src4_2), Ndef(\src5_2)], //rows (sources)
modList = [\mod1, \mod2, \mod3, \mod4, \mod5], //columns (modulations)
ampList = [\src1, \src2, \src3, \src4, \src5];
Tdef(\matrix_2, {
	var oldInd = 0, row, col, arr_old = ~array2D_2;
	loop {
		2.0.wait;
		"step".postln;

		~array2D_2.rowsDo({
			arg item, ind;


			item.do({
				arg item2, ind2;

				if (item2 != arr_old[ind, ind2],
					{//true
						Ndef(\mix_2).set(ampList[oldInd], 0.0);
						Ndef(\mix_2).set(ampList[ind], 0.9);
						oldInd = ind;


						row = ind;
						col = ind2;
						defList[ind].set(\azi, ~car2pol.(5, row, col)[0]);
						defList[ind].set(\ele, ~car2pol.(5, row, col)[1]);


						if (item2 == 1,
							{//true
								defList[ind].set(modList[ind2], 1);
							},
							{//false
								defList[ind].set(modList[ind2], 0);
							}
						);

						if (item.sum == 0,
							{//true
								Ndef(\mix_2).set(ampList[ind], 0.0);
							}
						);
					}
				);
			});
		});
		arr_old = ~array2D_2.deepCopy;

		//=======================================

		~boats1.do({
			arg boat, ind;

			boat.do({
				arg item2, ind2;

				if(item2 == [row, col],

					{//true
						"treffer".postln;
						boat[ind2] = 0;
						defList[row].set(\azi, (~car2pol.(5, row, col)[0]) + 360);

				});

				if(boat.sum == 0,

					{//true
						"schiff versenkt".postln;
						~boats1.removeAt(ind);

						fork{
							1.do({
								Ndef(\mix_2).set(ampList[0], 0.9);
								Ndef(\mix_2).set(ampList[1], 0.9);
								Ndef(\mix_2).set(ampList[2], 0.9);
								Ndef(\mix_2).set(ampList[3], 0.9);
								Ndef(\mix_2).set(ampList[4], 0.9);
								defList[0].set(\azi, ((340.rand) - 170));
								defList[1].set(\azi, ((340.rand) - 170));
								defList[2].set(\azi, ((340.rand) - 170));
								defList[3].set(\azi, ((340.rand) - 170));
								defList[4].set(\azi, ((340.rand) - 170));
								5.0.wait;
								Ndef(\mix_2).set(ampList[0], 0.0);
								Ndef(\mix_2).set(ampList[1], 0.0);
								Ndef(\mix_2).set(ampList[2], 0.0);
								Ndef(\mix_2).set(ampList[3], 0.0);
								Ndef(\mix_2).set(ampList[4], 0.0);

								Ndef(\mix_2).set(ampList[row], 0.9);
							});
						};
				});

				if(~boats1.sum == 0,

					{
						"field 2 won".postln;

						fork{
							1.do({
								Tdef(\matrix_1).stop;

								Ndef(\mix_2).set(ampList[0], 0.9);
								Ndef(\mix_2).set(ampList[1], 0.9);
								Ndef(\mix_2).set(ampList[2], 0.9);
								Ndef(\mix_2).set(ampList[3], 0.9);
								Ndef(\mix_2).set(ampList[4], 0.9);
								defList[0].set(\azi, ((340.rand) - 170));
								defList[1].set(\azi, ((340.rand) - 170));
								defList[2].set(\azi, ((340.rand) - 170));
								defList[3].set(\azi, ((340.rand) - 170));
								defList[4].set(\azi, ((340.rand) - 170));
								5.0.wait;
								defList[0].set(\azi, ((340.rand) - 170));
								defList[1].set(\azi, ((340.rand) - 170));
								defList[2].set(\azi, ((340.rand) - 170));
								defList[3].set(\azi, ((340.rand) - 170));
								defList[4].set(\azi, ((340.rand) - 170));

								defList[0].set(\ele, Rand.new(u, 90));
								defList[1].set(\ele, Rand.new(u, 90));
								defList[2].set(\ele, Rand.new(u, 90));
								defList[3].set(\ele, Rand.new(u, 90));
								defList[4].set(\ele, Rand.new(u, 90));
								5.0.wait;
								Ndef(\mix_1).set(ampList[0], 0.9);
								Ndef(\mix_1).set(ampList[1], 0.9);
								Ndef(\mix_1).set(ampList[2], 0.9);
								Ndef(\mix_1).set(ampList[3], 0.9);
								Ndef(\mix_1).set(ampList[4], 0.9);
								5.0.wait;
								Ndef(\mix_2).set(\amp, 0.0);

								7.0.wait;
								Ndef(\lowpad).stop(2);
								Ndef(\sprinkle).stop(2);
								Ndef(\bassy).stop(2);
								Ndef(\pad).stop(2);
								Ndef(\texture).stop(2);
								2.0.wait;
								Ndef(\mix_1).set(ampList[0], 0.0);
								Ndef(\mix_1).set(ampList[1], 0.0);
								Ndef(\mix_1).set(ampList[2], 0.0);
								Ndef(\mix_1).set(ampList[3], 0.0);
								Ndef(\mix_1).set(ampList[4], 0.0);
							});
						};
						Tdef(\matrix_2).stop;
				});
			});
		});
	}
});
)