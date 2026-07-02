import { Controller, Post, Body, Get, Put, Param } from '@nestjs/common';
import { QuotationService } from './quotation.service';

@Controller('quotation')
export class QuotationController {
  constructor(private readonly quotationService: QuotationService) {}

  @Post('process')
  async processPdf(@Body('filePath') filePath: string) {
    return this.quotationService.processPdfAndCompare(filePath);
  }

  @Get()
  async getQuotations() {
    return this.quotationService.getAllQuotations();
  }

  @Put(':id/correct')
  async correctAndRetrain(
    @Param('id') id: string,
    @Body('correctedData') correctedData: any,
  ) {
    return this.quotationService.correctAndRetrain(id, correctedData);
  }
}
