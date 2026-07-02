import { Model } from 'mongoose';
import { Quotation } from './schemas/quotation.schema';
export declare class QuotationService {
    private quotationModel;
    constructor(quotationModel: Model<Quotation>);
    processPdfAndCompare(filePath: string): Promise<any>;
    getAllQuotations(): Promise<Quotation[]>;
    correctAndRetrain(id: string, correctedData: any): Promise<any>;
}
